"""Find incidents in an imported table, explain them and put the answer in plain words."""

from bisect import bisect_left, bisect_right
from datetime import datetime
from itertools import accumulate
import math
import re
from statistics import fmean, median
from threading import Lock

MAX_BINS = 300
MAX_MEASURES = 8
MAX_SLICE_COLUMNS = 4
MAX_SLICE_VALUES = 20
MAX_CATEGORY_VALUES = 40
MIN_SLICE_ROWS = 30
MIN_ROWS_TO_SCAN = 8
START_Z = 5.0
EXTEND_Z = 2.5
MIN_PEAK_Z = 6.0
MIN_SLICE_PEAK_Z = 6.0
MIN_SLICE_INDICATOR_Z = 8.0
LONE_BIN_Z = 15.0
EXPLAIN_SAMPLE = 6000
EXPLAIN_EVENTS = 5
KEEP_SHARE = 0.5
COMPARABLE_SHARE = 0.7
CACHE_SIZE = 4
KIND_WORDS = (
    ("latency", ("latency", "response", "duration", "rtt", "p50", "p90", "p95", "p99")),
    ("queue", ("queue", "waiting", "backlog", "pending")),
    ("temperature", ("temp",)),
    ("sales", ("sales", "revenue", "orders", "amount", "price")),
    ("errors", ("error", "fail", "5xx")),
    ("wait", ("wait",)),
    ("load", ("cpu", "memory", "load", "util")),
)
PRIORITY = tuple(kind for kind, _ in KIND_WORDS)
OTHER_RANK = len(PRIORITY)
FLAG_RANK = OTHER_RANK + 1
QUESTION_KINDS = (
    ("latency", ("latency", "slow", "response time", "p95", "p99", "p50")),
    ("errors", ("error", "fail", "5xx", "crash")),
    ("temperature", ("temperature", "temp", "hot", "warm", "cold")),
    ("queue", ("queue", "backlog", "waiting")),
    ("sales", ("sales", "orders", "revenue")),
    ("load", ("cpu", "memory", "load")),
)
UNIT_SUFFIXES = (("_mb_s", "MB/s"), ("_ms", "ms"), ("_pct", "%"), ("_percent", "%"), ("_amps", "A"), ("_kmh", "km/h"), ("_c", "C"), ("_s", "s"))
STATUS_HINTS = ("status", "code", "result", "outcome", "state")
CODE_HINTS = ("status", "code", "class", "type", "region", "id")
ERROR_WORDS = ("error", "fail", "failed", "failure", "timeout", "exception", "declined", "denied", "5xx", "critical", "fatal")
TRUE_WORDS = ("1", "true", "yes", "y", "on", "open")
PLACE_WORDS = ("region", "zone", "site", "country", "city", "store", "location", "dc", "host", "node", "server")
WORD_FIXES = {"temp": "temperature", "db": "DB", "rtt": "RTT", "cpu": "CPU", "io": "IO"}
DIRECTION_WORDS = (
    ("average", ("average", "mean", "typical")),
    ("peak", ("peak", "max", "highest", "worst", "biggest")),
    ("now", ("now", "latest", "current", "right now", "last")),
    ("trend", ("trend", "growing", "rising", "falling", "drift", "increasing", "decreasing", "going up", "going down")),
)
CACHE = []
CACHE_LOCK = Lock()


def mad_scale(values):
    middle = median(values)
    return 1.4826 * median(abs(value - middle) for value in values)


def first_difference_noise(series):
    changes = [later - earlier for earlier, later in zip(series, series[1:]) if earlier is not None and later is not None]
    if len(changes) < 3:
        return 0.0
    middle = median(changes)
    return 1.4826 * median(abs(change - middle) for change in changes) / math.sqrt(2)


def number_text(value):
    magnitude = abs(value)
    if magnitude >= 1000:
        text = f"{value:,.0f}"
    elif magnitude >= 100:
        text = f"{value:.0f}"
    elif magnitude >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def multiple_text(ratio):
    return f"{ratio:.1f}x" if ratio < 10 else f"{ratio:.0f}x"


def join_words(parts):
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def capital(text):
    return text[:1].upper() + text[1:]


def split_unit(name):
    lowered = name.lower()
    for suffix, unit in UNIT_SUFFIXES:
        if lowered.endswith(suffix) and len(name) > len(suffix):
            return name[:-len(suffix)], unit
    return name, ""


def kind_of(name):
    lowered = name.lower()
    for kind, words in KIND_WORDS:
        if any(word in lowered for word in words):
            return kind
    return "other"


def readable_label(name, unit):
    base, _ = split_unit(name)
    words = [WORD_FIXES.get(word.lower(), word) for word in re.split(r"[_\s-]+", base) if word]
    if words and all(re.fullmatch(r"p\d+", word.lower()) for word in words) and unit == "ms":
        words.append("latency")
    return " ".join(words) or name


def make_measure(name, values, indicator=False, flag=False, label=None, source=None):
    base, unit = split_unit(name)
    kind = "errors" if indicator and not flag else kind_of(name)
    rank = FLAG_RANK if flag else PRIORITY.index(kind) if kind in PRIORITY else OTHER_RANK
    return dict(name=name, label=label or readable_label(name, unit), unit="" if indicator else unit, kind=kind, rank=rank,
                values=values, indicator=indicator, flag=flag, source=source)


def value_text(measure, value):
    if measure["indicator"]:
        return number_text(value * 100) + "%"
    text = number_text(value)
    unit = measure["unit"]
    if not unit:
        return text
    return text + unit if unit == "%" else f"{text} {unit}"


def delta_text(measure, value):
    text = number_text(abs(value))
    unit = measure["unit"]
    if measure["indicator"]:
        return number_text(abs(value) * 100) + " points"
    return text if not unit else text + unit if unit == "%" else f"{text} {unit}"


def uses_delta(measure, normal):
    return measure["kind"] == "temperature" or measure["unit"] == "C" or normal <= 0 and not measure["indicator"]


class Clock:
    def __init__(self, table, times):
        self.times = times
        self.kind = "row"
        self.scale = 1.0
        self.origin = 0.0
        self.whole_days = False
        self.cross_days = False
        if not times:
            return
        name = table["time"]
        if table["types"][name] == "time":
            self.kind = "clock"
        else:
            sample = times[:1000]
            steps = sorted(later - earlier for earlier, later in zip(sample, sample[1:]) if later > earlier)
            step = steps[len(steps) // 2] if steps else 1.0
            big = max(abs(times[0]), abs(times[-1]))
            if big >= 1e11:
                self.kind, self.scale = "clock", 0.001
            elif big >= 1e9:
                self.kind = "clock"
            else:
                self.kind = "relative"
                self.origin = times[0]
                self.scale = 0.001 if name.lower().endswith("ms") or step >= 100 else 1.0
        sample = times[:2000]
        steps = sorted(later - earlier for earlier, later in zip(sample, sample[1:]) if later > earlier)
        self.step = (steps[len(steps) // 2] if steps else 0.0) * self.scale
        self.span = (times[-1] - times[0]) * self.scale
        if self.kind == "clock":
            self.whole_days = all(self.moment(times[i]) is not None and self.moment(times[i]).strftime("%H%M%S") == "000000"
                                  for i in range(0, len(times), max(1, len(times) // 50)))
            first, last = self.moment(times[0]), self.moment(times[-1])
            self.cross_days = first is not None and last is not None and first.date() != last.date()

    def moment(self, raw):
        try:
            return datetime.fromtimestamp(raw * self.scale)
        except (ValueError, OverflowError, OSError):
            return None

    def label(self, position):
        if self.kind == "row":
            return f"row {position + 1}"
        raw = self.times[min(max(position, 0), len(self.times) - 1)]
        if self.kind == "relative":
            seconds = int(round((raw - self.origin) * self.scale))
            hours, rest = divmod(seconds, 3600)
            return f"t+{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"t+{rest // 60}:{rest % 60:02d}"
        moment = self.moment(raw)
        if moment is None:
            return f"{raw:g}"
        if self.whole_days:
            return moment.strftime("%Y-%m-%d")
        if self.cross_days:
            return moment.strftime("%b %d %H:%M")
        return moment.strftime("%H:%M:%S" if self.span < 180 else "%H:%M")

    def full_label(self, position):
        if self.kind == "clock" and not self.whole_days:
            moment = self.moment(self.times[min(max(position, 0), len(self.times) - 1)])
            return moment.strftime("%Y-%m-%d %H:%M") if moment is not None else self.label(position)
        return self.label(position)

    def duration(self, first, last):
        if self.kind == "row":
            return f"{last - first + 1:,} rows"
        seconds = (self.times[last] - self.times[first]) * self.scale
        if self.whole_days:
            days = int(round(seconds / 86400)) + 1
            return f"{days} day{'s' if days != 1 else ''}"
        if last == first:
            return "1 reading"
        seconds += self.step
        if seconds < 1:
            return f"{seconds * 1000:.0f} ms"
        if seconds < 90:
            return f"{seconds:.0f} s"
        minutes = seconds / 60
        if minutes < 90:
            return f"{minutes:.0f} min"
        hours = minutes / 60
        if hours < 48:
            whole, extra = int(hours), int(round(minutes - int(hours) * 60))
            return f"{whole} h {extra} min" if extra else f"{whole} h"
        return f"{hours / 24:.1f} days"


def classify_columns(table, count):
    categories, numeric = [], []
    for column in table["columns"]:
        if column == table["time"]:
            continue
        kind = table["types"][column]
        raw = table["data"][column]
        distinct = set(raw) - {None}
        if kind == "text":
            if 2 <= len(distinct) <= MAX_CATEGORY_VALUES and len(distinct) <= max(5, count // 4):
                categories.append((column, raw))
        elif kind == "number":
            binary = distinct <= {0.0, 1.0} and len(distinct) == 2
            coded = (any(hint in column.lower() for hint in CODE_HINTS) and 2 <= len(distinct) <= 12
                     and all(value == int(value) for value in distinct))
            if binary or coded:
                categories.append((column, [None if value is None else str(int(value)) for value in raw]))
            elif len(distinct) >= 2:
                numeric.append(column)
    return categories, numeric


def build_categories(entries, arrange):
    built = []
    for column, raw in entries:
        values = arrange([value if value is not None else "(blank)" for value in raw])
        names = sorted(set(values))
        lookup = {name: code for code, name in enumerate(names)}
        built.append(dict(name=column, names=names, codes=[lookup[value] for value in values],
                          binary=len(names) == 2 and set(name.lower() for name in names) <= set(TRUE_WORDS + ("0", "false", "no", "n", "off", "closed"))))
    return built


def derived_measures(categories, taken):
    found = []
    for category in categories:
        names = category["names"]
        lowered = category["name"].lower()
        if category["binary"]:
            truthy = {code for code, name in enumerate(names) if name.lower() in TRUE_WORDS}
            if len(truthy) == 1:
                values = [1.0 if code in truthy else 0.0 for code in category["codes"]]
                found.append(make_measure(category["name"], values, indicator=True, flag=True, label=readable_label(category["name"], ""), source=category["name"]))
                continue
        if any(hint in lowered for hint in STATUS_HINTS):
            bad = {code for code, name in enumerate(names) if re.fullmatch(r"5\d\d", name) or name.lower() in ERROR_WORDS}
            if bad and len(bad) < len(names):
                name = "errors" if "errors" not in taken else f"{category['name']} errors"
                found.append(make_measure(name, [1.0 if code in bad else 0.0 for code in category["codes"]], indicator=True, label="error rate", source=category["name"]))
                taken.add(name)
    return found


def prepare(table):
    count = table["rows"]
    times = order = None
    if table["time"] is not None:
        last = next((value for value in table["data"][table["time"]] if value is not None), 0.0)
        times = []
        for value in table["data"][table["time"]]:
            last = value if value is not None else last
            times.append(last)
        if any(later < earlier for earlier, later in zip(times, times[1:])):
            order = sorted(range(count), key=times.__getitem__)
            times = [times[i] for i in order]

    def arrange(values):
        return values if order is None else [values[i] for i in order]

    entries, numeric = classify_columns(table, count)
    categories = build_categories(entries, arrange)
    measures = []
    for column in numeric:
        raw = table["data"][column]
        present = [value for value in raw if value is not None]
        fill = median(present)
        measures.append(make_measure(column, arrange([fill if value is None else value for value in raw]), source=column))
        measures[-1]["present"] = len(present)
    if not measures:
        raise ValueError("This file has no numeric columns to analyze.")
    measures.sort(key=lambda measure: measure["rank"])
    measures = measures[:MAX_MEASURES]
    for measure in measures:
        measure.setdefault("present", count)
    derived = derived_measures(categories, {measure["name"] for measure in measures})
    for measure in derived:
        measure["present"] = count
    measures = sorted(measures + derived, key=lambda measure: measure["rank"])
    return dict(count=count, times=times, clock=Clock(table, times), categories=categories, measures=measures,
                edges=bin_edges(count, times))


def bin_edges(count, times=None):
    bins = min(count, MAX_BINS)
    edges = [index * count // bins for index in range(bins + 1)]
    if times is None:
        return edges
    snapped = {0, count}
    for edge in edges[1:-1]:
        if times[edge] == times[edge - 1]:
            edge = bisect_right(times, times[edge])
        snapped.add(edge)
    return sorted(snapped)


def prefix_sums(values):
    return list(accumulate(values, initial=0.0))


def bin_means(prefix, edges):
    return [(prefix[end] - prefix[start]) / (end - start) for start, end in zip(edges, edges[1:])]


def rolling_baseline(series):
    size = len(series)
    window = max(9, int(size * 0.6))
    half = window // 2
    step = max(1, half // 3)
    knots = list(range(0, size, step))
    if knots[-1] != size - 1:
        knots.append(size - 1)
    medians = []
    for knot in knots:
        segment = [value for value in series[max(0, knot - half):knot + half + 1] if value is not None]
        medians.append(median(segment) if segment else None)
    known = [(knot, value) for knot, value in zip(knots, medians) if value is not None]
    if not known:
        return [0.0] * size
    baseline, cursor = [], 0
    for index in range(size):
        while cursor + 1 < len(known) and known[cursor + 1][0] <= index:
            cursor += 1
        left = known[cursor]
        right = known[min(cursor + 1, len(known) - 1)]
        if right[0] == left[0] or index <= left[0]:
            baseline.append(left[1])
        else:
            baseline.append(left[1] + (right[1] - left[1]) * (index - left[0]) / (right[0] - left[0]))
    return baseline


def stretches_in(series, baseline, scale):
    size = len(series)
    z = [0.0 if value is None else (value - base) / scale for value, base in zip(series, baseline)]
    found, index = [], 0
    while index < size:
        if abs(z[index]) >= START_Z:
            sign = 1 if z[index] > 0 else -1
            first = last = index
            while first > 0 and z[first - 1] * sign >= EXTEND_Z:
                first -= 1
            while last + 1 < size and z[last + 1] * sign >= EXTEND_Z:
                last += 1
            found.append([first, last, sign])
            index = last + 1
        else:
            index += 1
    merged = []
    for first, last, sign in found:
        if merged and merged[-1][2] == sign and first - merged[-1][1] - 1 <= max(2, math.ceil(0.25 * max(last - first + 1, merged[-1][1] - merged[-1][0] + 1))):
            merged[-1][1] = last
        else:
            merged.append([first, last, sign])
    return merged, z


def series_incidents(series, level, floor_extra, measure, slice_key, minimum_change=0.0):
    present = [value for value in series if value is not None]
    if len(present) < max(MIN_ROWS_TO_SCAN, len(series) * 0.6):
        return []
    baseline = rolling_baseline(series)
    residual = [value - base for value, base in zip(series, baseline) if value is not None]
    scale = max(mad_scale(residual), 0.02 * abs(level), 0.02 * (max(present) - min(present)), floor_extra, 1e-9)
    stretches, z = stretches_in(series, baseline, scale)
    weight = 0.35 if measure["flag"] else 1.0
    incidents = []
    for first, last, sign in stretches:
        peak = max(range(first, last + 1), key=lambda index: sign * z[index])
        deviation = series[peak] - baseline[peak]
        needed = (MIN_SLICE_INDICATOR_Z if measure["indicator"] else MIN_SLICE_PEAK_Z) if slice_key is not None else MIN_PEAK_Z
        if sign * z[peak] < needed or (last == first and (sign * z[peak] < LONE_BIN_Z or measure["indicator"])):
            continue
        if slice_key is None and abs(deviation) < 0.1 * abs(baseline[peak]):
            continue
        if slice_key is not None and abs(deviation) < minimum_change:
            continue
        relative = min(abs(deviation) / max(abs(baseline[peak]), 4 * scale), 8.0)
        score = math.log1p(relative) * (1 + math.log10(1 + last - first + 1)) * weight
        incidents.append(dict(measure=measure["name"], slice=slice_key, first=first, last=last, sign=sign, score=score,
                              z=sign * z[peak], peak=peak, scale=scale, level=baseline[peak]))
    return incidents


def indicator_floor(values_mean, rows_per_bin):
    return math.sqrt(max(values_mean * (1 - values_mean), 1e-6) / max(rows_per_bin, 1.0))


def find_drift(series, measure, noise_floor):
    size = len(series)
    if size < 24:
        return None
    half = size // 2
    slopes = [(series[index + half] - series[index]) / half for index in range(size - half)]
    middle = median(slopes)
    if middle == 0:
        return None
    agreement = sum(1 for slope in slopes if slope * middle > 0) / len(slopes)
    change = middle * (size - 1)
    noise = max(first_difference_noise(series), noise_floor, 1e-9)
    edge = max(2, size // 10)
    start_value, end_value = median(series[:edge]), median(series[-max(2, size // 30):])
    level = median(series)
    if agreement < 0.85 or abs(change) < 8 * noise or abs(end_value - start_value) < 6 * noise:
        return None
    if abs(end_value - start_value) < 0.03 * abs(level) and abs(end_value - start_value) < 20 * noise:
        return None
    return dict(measure=measure["name"], onset=onset_of(series, start_value, noise), start_value=start_value, end_value=end_value,
                noise=noise, level=level, sign=1 if end_value > start_value else -1)


def onset_of(series, start_value, noise):
    size = len(series)
    run = max(3, size // 30)
    end_value = median(series[-max(2, size // 30):])
    sign = 1 if end_value > start_value else -1
    for index in range(size - run):
        if all(sign * (series[index + offset] - start_value) > 4 * noise for offset in range(run)):
            slope = sign * (end_value - series[index]) / max(size - 1 - index, 1)
            return max(0, index - int(round(4 * noise / slope))) if slope > 0 else index
    return 0


def sibling_pairs(measures):
    pairs = []
    plain = [measure for measure in measures if not measure["indicator"]]
    for position, first in enumerate(plain):
        first_tokens = first["name"].lower().split("_")
        for second in plain[position + 1:]:
            second_tokens = second["name"].lower().split("_")
            if len(first_tokens) != len(second_tokens):
                continue
            different = [(a, b) for a, b in zip(first_tokens, second_tokens) if a != b]
            if len(different) == 1 and all(len(token) <= 2 or re.fullmatch(r"[a-z]?\d+", token) for token in different[0]):
                pairs.append((first, second))
    return pairs


def find_divergence(first, second, first_series, second_series):
    gap = [a - b for a, b in zip(first_series, second_series)]
    size = len(gap)
    if size < 24:
        return None
    edge = max(2, size // 8)
    early, late = median(gap[:edge]), median(gap[-max(2, size // 30):])
    noise = max(first_difference_noise(gap), 1e-9)
    level = (median(first_series) + median(second_series)) / 2
    if abs(late - early) < 10 * noise or abs(late - early) < 0.03 * abs(level):
        return None
    onset = onset_of(gap, early, noise)
    moved_first = abs(median(first_series[-edge:]) - median(first_series[:edge]))
    moved_second = abs(median(second_series[-edge:]) - median(second_series[:edge]))
    mover, other = (first, second) if moved_first >= moved_second else (second, first)
    return dict(mover=mover["name"], other=other["name"], onset=onset, early=early, late=late, noise=noise, level=level,
                sign=1 if late > early else -1, first=first["name"], second=second["name"])


def slice_groups(frame):
    groups = []
    for category in frame["categories"][:]:
        if category["binary"] or not 2 <= len(category["names"]) <= MAX_SLICE_VALUES:
            continue
        buckets = {}
        for position, code in enumerate(category["codes"]):
            bucket = buckets.get(code)
            if bucket is None:
                buckets[code] = [position]
            else:
                bucket.append(position)
        usable = {code: rows for code, rows in buckets.items() if len(rows) >= max(MIN_SLICE_ROWS, 0.01 * frame["count"])}
        if len(usable) >= 2:
            groups.append((category, usable))
        if len(groups) >= MAX_SLICE_COLUMNS:
            break
    return groups


def slice_incidents(frame, measure, prefix, edges, groups, level):
    incidents = []
    values = measure["values"]
    bins = len(edges) - 1
    normalized = not measure["indicator"] and level > 0 and min(values) >= 0
    getter = values.__getitem__
    for category_index, (category, buckets) in enumerate(groups):
        for code, rows in buckets.items():
            cuts = [bisect_left(rows, edge) for edge in edges]
            series = []
            for index in range(bins):
                lower, upper = max(0, index - 1), min(bins, index + 2)
                inside = cuts[upper] - cuts[lower]
                outside = edges[upper] - edges[lower] - inside
                if inside == 0 or outside == 0:
                    series.append(None)
                    continue
                part = sum(map(getter, rows[cuts[lower]:cuts[upper]]))
                rest = (prefix[edges[upper]] - prefix[edges[lower]] - part) / outside
                mean = part / inside
                if normalized and rest < 0.5 * level:
                    series.append(None)
                    continue
                series.append((mean - rest) / max(rest, 0.1 * level) if normalized else mean - rest)
            rows_per_bin = len(rows) / bins
            floor = indicator_floor(fmean(values), rows_per_bin) if measure["indicator"] else 0.02 if normalized else 0.0
            incidents.extend(series_incidents(series, 0.0 if normalized else level, floor, measure, (category_index, code),
                                              0.15 if normalized else 0.1 * abs(level) if not measure["indicator"] else 0.05))
    return incidents


def cluster_events(incidents, measures_by_name):
    events = []
    for incident in sorted(incidents, key=lambda item: -item["score"]):
        for event in events:
            if incident["first"] <= event["last"] + 2 and incident["last"] >= event["first"] - 2:
                event["members"].append(incident)
                break
        else:
            events.append(dict(first=incident["first"], last=incident["last"], members=[incident]))
    for event in events:
        ranked = sorted(event["members"], key=lambda item: (measures_by_name[item["measure"]]["rank"], -item["score"]))
        lead = ranked[0]
        event.update(lead=lead, first=lead["first"], last=lead["last"], score=max(item["score"] for item in event["members"]))
    return sorted(events, key=lambda event: -event["score"])


def explain_event(frame, edges, event, measures_by_name):
    count = frame["count"]
    lead = event["lead"]
    measure = measures_by_name[lead["measure"]]
    low, high = edges[lead["first"]], edges[lead["last"] + 1]
    margin = max(1, count // (len(edges) - 1))
    stride = max(1, count // EXPLAIN_SAMPLE)
    inside_rows = list(range(low, high))
    outside_rows = [position for position in range(0, count, stride) if position < low - margin or position >= high + margin]
    sign = lead["sign"]
    values = measure["values"]
    outside_values = [values[position] for position in outside_rows]
    result = dict(culprit=[], constraints=[], share=None, inside=inside_rows, outside=outside_rows)
    if not inside_rows or not outside_rows:
        return result
    row_scale = max(mad_scale(outside_values), 1e-9)
    if measure["indicator"]:
        rate = fmean(outside_values)
        row_scale = max(row_scale, math.sqrt(max(rate * (1 - rate), 1e-6)))
    overall_out = fmean(outside_values)
    window_gain = sign * sum(values[position] - overall_out for position in inside_rows)
    current_inside, current_outside = inside_rows, outside_rows
    current_relative = window_gain / len(inside_rows) / max(abs(overall_out), row_scale)
    used = set()
    captured = 1.0
    for _ in range(3):
        best = None
        for index, category in enumerate(frame["categories"]):
            if index in used or category["binary"] or len(category["names"]) < 2:
                continue
            codes = category["codes"]
            inside_count, inside_sum, outside_count, outside_sum = {}, {}, {}, {}
            for position in current_inside:
                code = codes[position]
                inside_count[code] = inside_count.get(code, 0) + 1
                inside_sum[code] = inside_sum.get(code, 0.0) + values[position]
            for position in current_outside:
                code = codes[position]
                outside_count[code] = outside_count.get(code, 0) + 1
                outside_sum[code] = outside_sum.get(code, 0.0) + values[position]
            fallback = fmean([values[position] for position in current_outside])
            losses = {}
            for code, number in inside_count.items():
                baseline = outside_sum[code] / outside_count[code] if code in outside_count else fallback
                losses[code] = (sign * (inside_sum[code] - number * baseline), baseline)
            positive = sum(loss for loss, _ in losses.values() if loss > 0)
            if positive <= 0 or len(inside_count) < 2:
                continue
            for code, (loss, baseline) in losses.items():
                number = inside_count[code]
                if number < 5 or number > 0.9 * len(current_inside) or loss / positive < 0.6:
                    continue
                relative = loss / number / max(abs(baseline), row_scale)
                if relative >= 1.3 * current_relative and (best is None or relative > best[0]):
                    best = (relative, index, code, loss / positive)
        if best is None:
            break
        current_relative, index, code, capture = best
        captured *= capture
        codes = frame["categories"][index]["codes"]
        current_inside = [position for position in current_inside if codes[position] == code]
        current_outside = [position for position in current_outside if codes[position] == code]
        used.add(index)
        result["culprit"].append((frame["categories"][index]["name"], frame["categories"][index]["names"][code]))
        result["constraints"].append((frame["categories"][index], code))
    if result["culprit"]:
        result["inside"], result["outside"] = current_inside, current_outside
        result["share"] = min(1.0, captured)
    return result


def culprit_text(culprit):
    parts = []
    for column, value in culprit:
        bare = value.startswith("/") or (value.isalpha() and len(value) >= 3) or "-" in value or " " in value
        words = re.sub(r"[_\s]+id$", "", column.replace("_", " "), flags=re.I)
        parts.append((any(word in column.lower() for word in PLACE_WORDS), value if bare else f"{words} {value}"))
    parts.sort(key=lambda part: part[0])
    names = [text for _, text in parts]
    if len(names) == 1:
        return names[0]
    return names[0] + "".join((" in " if place else " and ") + text for (place, text) in parts[1:])


def moved_with(frame, lead_name, inside, outside, related):
    moved = []
    if len(inside) < 3 or len(outside) < 5:
        return moved
    for measure in frame["measures"]:
        if measure["name"] == lead_name:
            continue
        values = measure["values"]
        inside_values = [values[position] for position in inside]
        outside_values = [values[position] for position in outside]
        if measure["indicator"]:
            now = fmean(inside_values)
            normal = fmean(outside_values)
            scale = math.sqrt(max(normal * (1 - normal), 1e-6))
        else:
            now = median(inside_values)
            normal = median(outside_values)
            scale = max(mad_scale(outside_values), 0.02 * abs(normal), 1e-9)
        z = abs(now - normal) / (scale / math.sqrt(len(inside_values)))
        effect = abs(now - normal) / max(abs(normal), scale)
        member = measure["name"] in related
        if z < (0 if member else 6) or effect < (0.25 if member else 0.75):
            continue
        moved.append(dict(measure=measure, now=now, normal=normal, effect=effect, up=now > normal))
    moved.sort(key=lambda item: -item["effect"])
    return moved[:3]


def change_phrase(item):
    measure = item["measure"]
    return f"{measure['label']} went from {value_text(measure, item['normal'])} to {value_text(measure, item['now'])}"


def peak_and_normal(measure, inside, outside, sign):
    values = measure["values"]
    size = max(20, len(inside) // 8) if measure["indicator"] else 5
    chunks = [inside[index:index + size] for index in range(0, len(inside), size)] or [inside]
    means = [fmean([values[position] for position in chunk]) for chunk in chunks if chunk]
    peak = max(means) if sign > 0 else min(means)
    position = chunks[means.index(peak)][0] if chunks and chunks[0] else 0
    outside_values = [values[index] for index in outside]
    normal = (fmean(outside_values) if measure["indicator"] else median(outside_values)) if outside_values else peak
    return peak, normal, position


def subset_rows(frame, constraints, low, high):
    codes = [(category["codes"], code) for category, code in constraints]
    return [position for position in range(max(0, low), min(frame["count"], high)) if all(column[position] == code for column, code in codes)]


def abnormal_span(values, rows, threshold, sign):
    flags = [sign * (values[position] - threshold) >= 0 for position in rows]
    first = next((index for index, flag in enumerate(flags) if flag and sum(flags[index:index + 3]) >= min(2, len(flags) - index)), None)
    last = next((index for index in range(len(flags) - 1, -1, -1) if flags[index] and sum(flags[max(0, index - 2):index + 1]) >= min(2, index + 1)), None)
    if first is None or last is None or last < first:
        return None
    return first, last


def stayed_normal(frame, measure, constraints, low, high, change):
    sentences = []
    values = measure["values"]
    for index in reversed(range(len(constraints))):
        if len(sentences) >= 2:
            break
        category, code = constraints[index]
        others = [(other, other_code) for position, (other, other_code) in enumerate(constraints) if position != index]
        columns = [(other["codes"], other_code) for other, other_code in others]
        alternative = category["codes"]
        inside = [position for position in range(low, high) if alternative[position] != code and all(column[position] == value for column, value in columns)]
        span = high - low
        outside = [position for position in list(range(max(0, low - span), low)) + list(range(high, min(frame["count"], high + span)))
                   if alternative[position] != code and all(column[position] == value for column, value in columns)]
        if len(inside) < 5 or len(outside) < 5:
            continue
        gap = abs(fmean([values[position] for position in inside]) - fmean([values[position] for position in outside]))
        if gap >= 0.2 * abs(change):
            continue
        names = sorted({category["names"][alternative[position]] for position in inside})
        if len(names) <= 2:
            alternative_name = " and ".join(names)
        else:
            words = re.sub(r"[_\s]+id$", "", category["name"].replace("_", " "), flags=re.I)
            alternative_name = f"other {words}" + ("" if words.endswith("s") else "s")
        pairs = [(constraint[0]["name"], constraint[0]["names"][constraint[1]] if position != index else alternative_name)
                 for position, constraint in enumerate(constraints)]
        sentences.append(culprit_text(pairs))
    return sentences


def describe_event(frame, edges, event, measures_by_name, explained):
    clock = frame["clock"]
    lead = event["lead"]
    measure = measures_by_name[lead["measure"]]
    sign = lead["sign"]
    values = measure["values"]
    inside, outside = explained["inside"], explained["outside"]
    peak, normal, peak_position = peak_and_normal(measure, inside, outside, sign)
    first_position, last_position = edges[lead["first"]], edges[lead["last"] + 1] - 1
    bin_size = max(1, frame["count"] // (len(edges) - 1))
    constraints = explained["constraints"]
    rows = subset_rows(frame, constraints, first_position - 2 * bin_size, last_position + 2 * bin_size + 1)
    span = abnormal_span(values, rows, normal + 0.5 * (peak - normal), sign) if rows else None
    if span is not None:
        first_position, last_position = rows[span[0]], rows[span[1]]
        refined = [position for position in rows if first_position <= position <= last_position]
        if len(refined) >= 3:
            inside = refined
            peak, normal, peak_position = peak_and_normal(measure, inside, outside, sign)
    related = {member["measure"] for member in event["members"]}
    moved = moved_with(frame, measure["name"], inside, outside, related)
    onset, cause = first_position, None
    wide = subset_rows(frame, constraints, first_position - max(last_position - first_position, 2 * bin_size) // 2, first_position + 1)
    wide_rows = wide + [position for position in inside if position > first_position]
    for item in moved:
        found = abnormal_span(item["measure"]["values"], wide_rows, item["normal"] + 0.5 * (item["now"] - item["normal"]), 1 if item["up"] else -1) if wide_rows else None
        item["start"] = wide_rows[found[0]] if found else first_position
        eligible = item["measure"]["kind"] not in (measure["kind"], "errors") and (item["start"] <= first_position or item["measure"]["flag"])
        if eligible and (cause is None or item["start"] < cause["start"]):
            cause = item
    culprit = culprit_text(explained["culprit"]) if explained["culprit"] else ""
    scope = f" on {culprit}" if culprit else ""
    after = min(bisect_right(frame["times"], frame["times"][last_position]), frame["count"] - 1) if frame["times"] else last_position
    start, end = clock.label(first_position), clock.label(after)
    when = f"between {start} and {end}" if start != end else f"at {start}"
    sentences = [f"{capital(measure['label'])} went from {value_text(measure, normal)} to {value_text(measure, peak)}{scope} {when}."]
    others = [item for item in moved if item is not cause]
    if cause is not None and sum(1 for position in wide_rows if cause["start"] <= position < first_position) >= 3:
        onset = cause["start"]
        sentences.append(f"{capital(change_phrase(cause))} at {clock.label(onset)}, then {measure['label']} {'rose' if sign > 0 else 'fell'}.")
    elif cause is not None:
        sentences.append(f"Likely cause: {cause['measure']['label']}, which went from {value_text(cause['measure'], cause['normal'])} to {value_text(cause['measure'], cause['now'])} at the same moment.")
    quiet = stayed_normal(frame, measure, constraints, first_position, last_position + 1, peak - normal) if constraints else []
    if quiet:
        sentences.append(f"{capital(measure['label'])} stayed normal for {' and for '.join(quiet)}.")
    if others:
        sentences.append(f"{capital(join_words([change_phrase(item) for item in others]))} at the same time.")
    if culprit and explained["share"] is not None:
        sentences.append(f"{round(explained['share'] * 100)}% of the {'extra' if sign > 0 else 'lost'} {measure['label']} came from {culprit}.")
    flag = next((item["measure"]["label"] for item in moved if item["measure"]["flag"]), "")
    subject = f"{culprit} " if culprit else ""
    smaller = f"{subject}{measure['label']} went from {value_text(measure, normal)} to {value_text(measure, peak)} at {start}"
    smaller += f" during {flag} periods, which looks like normal handling." if flag else "."
    return dict(kind="incident", measure=measure["name"], label=measure["label"], sign=sign, score=event["score"],
                first=onset, last=last_position, peak_position=peak_position, start=clock.label(onset), end=end,
                duration=clock.duration(onset, last_position), peak=value_text(measure, peak), normal=value_text(measure, normal),
                peak_raw=peak, normal_raw=normal, culprit=culprit, share=explained["share"], gap="", sentences=sentences,
                moved=[change_phrase(item) for item in moved], moved_items=moved, smaller=smaller,
                members=sorted({member["measure"] for member in event["members"]}), rows_inside=len(inside))


def drift_event(frame, drift, measures_by_name, bins):
    clock, count = frame["clock"], frame["count"]
    measure = measures_by_name[drift["measure"]]
    edges = frame["edges"]
    first_position = edges[drift["onset"]]
    last_position = count - 1
    direction = "up" if drift["sign"] > 0 else "down"
    change = abs(drift["end_value"] - drift["start_value"])
    sentence = (f"{measure['name']} is drifting {direction}: {value_text(measure, drift['start_value'])} at the start, "
                f"{value_text(measure, drift['end_value'])} now, steady since {clock.label(first_position)}.")
    relative = min(change / max(abs(drift["level"]), 4 * drift["noise"]), 8.0)
    return dict(kind="drift", measure=measure["name"], label=measure["label"], sign=drift["sign"],
                score=math.log1p(relative) * (1 + math.log10(1 + bins)), first=first_position, last=last_position, peak_position=last_position,
                start=clock.label(first_position), end=clock.label(last_position), duration=clock.duration(first_position, last_position),
                peak=value_text(measure, drift["end_value"]), normal=value_text(measure, drift["start_value"]), culprit="", share=None,
                gap="", sentences=[sentence], moved=[], moved_items=[], members=[measure["name"]], rows_inside=count - first_position,
                peak_raw=drift["end_value"], normal_raw=drift["start_value"], smaller=sentence)


def divergence_event(frame, found, measures_by_name, bins):
    clock, count = frame["clock"], frame["count"]
    mover = measures_by_name[found["mover"]]
    edges = frame["edges"]
    first_position = edges[found["onset"]]
    last_position = count - 1
    gap_now, gap_before = abs(found["late"]), abs(found["early"])
    unit = f" {mover['unit']}" if mover["unit"] and mover["unit"] != "%" else mover["unit"]
    sentence = (f"{found['mover']} is drifting away from {found['other']} since {clock.label(first_position)}. "
                f"They were {number_text(gap_before)}{unit} apart and are now {number_text(gap_now)}{unit} apart.")
    relative = min(abs(found["late"] - found["early"]) / max(abs(found["level"]), 4 * found["noise"]), 8.0)
    return dict(kind="divergence", measure=found["mover"], label=f"{found['mover']} vs {found['other']}", sign=found["sign"],
                score=math.log1p(relative) * (1 + math.log10(1 + bins)) * 1.2, first=first_position, last=last_position, peak_position=last_position,
                start=clock.label(first_position), end=clock.label(last_position), duration=clock.duration(first_position, last_position),
                peak=f"{number_text(gap_now)}{unit} apart", normal=f"{number_text(gap_before)}{unit} apart", culprit="", share=None, gap="",
                sentences=[sentence], moved=[], moved_items=[], members=[found["mover"], found["other"]], rows_inside=count - first_position,
                peak_raw=gap_now, normal_raw=gap_before, smaller=sentence)


def analyze(table):
    frame = prepare(table)
    count = frame["count"]
    measures = frame["measures"]
    measures_by_name = {measure["name"]: measure for measure in measures}
    for measure in measures:
        if measure["kind"] == "sales" and not measure["unit"] and not measure["indicator"]:
            measure["unit"] = "per hour" if 3300 <= frame["clock"].step <= 3900 else "per row"
    result = dict(frame=frame, events=[], measures_by_name=measures_by_name, edges=None, drifting=set())
    if count < MIN_ROWS_TO_SCAN:
        return result
    edges = frame["edges"]
    bins = len(edges) - 1
    result["edges"] = edges
    prefixes, series_by_name, incidents, drifts = {}, {}, [], []
    for measure in measures:
        prefix = prefix_sums(measure["values"])
        prefixes[measure["name"]] = prefix
        series = bin_means(prefix, edges)
        series_by_name[measure["name"]] = series
        level = median(series)
        floor = indicator_floor(fmean(measure["values"]), count / bins) if measure["indicator"] else 0.0
        drift = None if measure["indicator"] else find_drift(series, measure, floor)
        if drift is not None:
            drifts.append(drift)
            result["drifting"].add(measure["name"])
            continue
        incidents.extend(series_incidents(series, level, floor, measure, None))
    groups = slice_groups(frame)
    if groups:
        for measure in measures:
            if measure["name"] not in result["drifting"]:
                level = median(series_by_name[measure["name"]])
                incidents.extend(slice_incidents(frame, measure, prefixes[measure["name"]], edges, groups, level))
    events = cluster_events(incidents, measures_by_name)[:EXPLAIN_EVENTS + 3]
    described = []
    for number, event in enumerate(events):
        explained = explain_event(frame, edges, event, measures_by_name) if number < EXPLAIN_EVENTS else dict(culprit=[], constraints=[], share=None, inside=list(range(edges[event["first"]], edges[event["last"] + 1])), outside=[])
        if not explained["outside"]:
            stride = max(1, count // EXPLAIN_SAMPLE)
            low, high = edges[event["first"]], edges[event["last"] + 1]
            explained["outside"] = [position for position in range(0, count, stride) if position < low or position >= high]
        described.append(describe_event(frame, edges, event, measures_by_name, explained))
    pair_names = set()
    for first, second in sibling_pairs(measures):
        found = find_divergence(first, second, series_by_name[first["name"]], series_by_name[second["name"]])
        if found is not None:
            pair_names.update((first["name"], second["name"]))
            described.append(divergence_event(frame, found, measures_by_name, bins))
    for drift in drifts:
        if drift["measure"] not in pair_names:
            described.append(drift_event(frame, drift, measures_by_name, bins))
    described.sort(key=lambda event: -event["score"])
    result["events"] = [event for event in described if event["score"] >= KEEP_SHARE * described[0]["score"]]
    return result


def cached_analysis(table):
    with CACHE_LOCK:
        for entry in CACHE:
            if entry[0] is table:
                return entry[1]
    analysis = analyze(table)
    with CACHE_LOCK:
        CACHE.insert(0, (table, analysis))
        del CACHE[CACHE_SIZE:]
    return analysis


def pick_measure(analysis, question):
    lowered = question.lower()
    measures = analysis["frame"]["measures"]
    for measure in measures:
        if measure["name"].lower() in lowered or (len(measure["label"]) > 3 and measure["label"].lower() in lowered):
            return measure
    for kind, words in QUESTION_KINDS:
        if any(re.search(r"\b" + re.escape(word), lowered) for word in words):
            matching = [measure for measure in measures if measure["kind"] == kind]
            if matching:
                scored = {event["measure"]: event["score"] for event in reversed(analysis["events"])}
                return max(matching, key=lambda measure: scored.get(measure["name"], 0))
    return None


def question_mode(lowered):
    for mode, words in DIRECTION_WORDS:
        if any(re.search(r"\b" + re.escape(word), lowered) for word in words):
            return mode
    return "story"


def reading_stats(analysis, measure):
    values = measure["values"]
    clock = analysis["frame"]["clock"]
    top = max(range(len(values)), key=values.__getitem__)
    low = min(range(len(values)), key=values.__getitem__)
    recent = max(1, len(values) // 10)
    change = fmean(values[-recent:]) - fmean(values[-2 * recent:-recent] or values[:recent])
    spread = mad_scale(values[::max(1, len(values) // 2000)]) or 1e-9
    direction = "rising" if change > spread * 0.25 else "falling" if change < -spread * 0.25 else "steady"
    return dict(mean=fmean(values), top=top, low=low, last=len(values) - 1, direction=direction, clock=clock)


def plain_value(measure, value):
    if measure["indicator"]:
        return value_text(measure, value)
    unit = measure["unit"]
    text = f"{value:g}"
    return text if not unit else text + unit if unit == "%" else f"{text} {unit}"


def lookup_answer(analysis, measure, mode):
    stats = reading_stats(analysis, measure)
    clock = stats["clock"]
    present = measure["present"]
    if mode == "average":
        return f"{capital(measure['label'])} averages {stats['mean']:.2f} over {present:,} readings."
    if mode == "peak":
        return f"{capital(measure['label'])} peaked at {plain_value(measure, measure['values'][stats['top']])} at {clock.label(stats['top'])}."
    if mode == "now":
        return f"{capital(measure['label'])} is {plain_value(measure, measure['values'][stats['last']])} at {clock.label(stats['last'])}, {stats['direction']}."
    return f"{capital(measure['label'])} is {stats['direction']} and now {plain_value(measure, measure['values'][stats['last']])}."


def quiet_sentences(analysis, measure):
    stats = reading_stats(analysis, measure)
    values = measure["values"]
    return [f"Nothing unusual in {measure['label']}: it stays near {value_text(measure, median(values))}.",
            f"It ranged from {value_text(measure, values[stats['low']])} to {value_text(measure, values[stats['top']])} and is {stats['direction']} now."]


def coverage_line(analysis, table):
    frame = analysis["frame"]
    clock = frame["clock"]
    last = frame["count"] - 1
    return f"{table['rows']:,} rows from {clock.full_label(0)} to {clock.full_label(last)}."


def event_rows(events):
    rows = []
    for event in events[:8]:
        signal = event["label"] if event["kind"] != "drift" else f"{event['measure']} drift"
        top = f"{event['culprit']} ({round(event['share'] * 100)}%)" if event["culprit"] and event["share"] is not None else event["culprit"] or "none"
        rows.append([event["start"], event["end"], event["duration"], signal, event["peak"], event["normal"],
                     ", ".join(event["moved"]) or "none", top])
    return rows


def transcript_events(analysis, events, measure, table):
    frame = analysis["frame"]
    clock = frame["clock"]
    last = frame["count"] - 1
    lines = [(0, "start", f"{frame['count']:,} rows begin")]
    for event in events[:3]:
        if event["kind"] == "incident":
            word = "spike" if event["sign"] > 0 else "drop"
            scope = f" on {event['culprit']}" if event["culprit"] else ""
            lines.append((event["first"], word, f"{event['label']} leaves its normal {event['normal']}{scope}"))
            lines.append((event["peak_position"], "peak" if event["sign"] > 0 else "low", f"{event['label']} reaches {event['peak']}"))
            lines.append((event["last"], "calm", f"{event['label']} back to normal"))
        elif event["kind"] == "drift":
            lines.append((event["first"], "drift", f"{event['measure']} starts drifting {'up' if event['sign'] > 0 else 'down'} from {event['normal']}"))
        else:
            lines.append((event["first"], "split", event["sentences"][0].split(" since ")[0]))
    stats = reading_stats(analysis, measure)
    lines.append((last, "now", f"{measure['label']} {value_text(measure, measure['values'][last])}, {stats['direction']}"))
    lines.sort(key=lambda line: line[0])
    return [(clock.label(position), kind, note) for position, kind, note in lines]



def readings_text(analysis, table, lead_event, measure):
    parts = [coverage_line(analysis, table)]
    if lead_event is not None and lead_event["kind"] == "incident":
        parts.append(f"{capital(lead_event['label'])}: normal {lead_event['normal']}, peak {lead_event['peak']}, "
                     f"{lead_event['rows_inside']:,} rows in the window from {lead_event['start']} to {lead_event['end']}.")
        if lead_event["culprit"] and lead_event["share"] is not None:
            parts.append(f"{round(lead_event['share'] * 100)}% of the change came from {lead_event['culprit']}.")
        for item in lead_event["moved_items"]:
            parts.append(f"{capital(item['measure']['label'])}: normal {value_text(item['measure'], item['normal'])}, in the window {value_text(item['measure'], item['now'])}.")
    elif lead_event is not None:
        parts.append(f"{capital(lead_event['label'])}: {lead_event['normal']} early, {lead_event['peak']} now.")
    else:
        stats = reading_stats(analysis, measure)
        values = measure["values"]
        parts.append(f"{capital(measure['label'])}: average {stats['mean']:.2f}, highest {value_text(measure, values[stats['top']])}, latest {value_text(measure, values[stats['last']])}.")
    return " ".join(parts)


def build_answer(table, question, style):
    analysis = cached_analysis(table)
    lowered = question.lower()
    measure = pick_measure(analysis, question)
    mode = question_mode(lowered)
    events = analysis["events"]
    focus_events = events if measure is None else [event for event in events if measure["name"] in event["members"]]
    story_mode = mode == "story" or (mode == "trend" and focus_events)
    if measure is None:
        measure = analysis["measures_by_name"][focus_events[0]["measure"]] if focus_events and focus_events[0]["measure"] in analysis["measures_by_name"] \
            else analysis["frame"]["measures"][0]
    if mode == "trend":
        drifts = [event for event in focus_events if event["kind"] != "incident"]
        focus_events = drifts or focus_events
    lead = focus_events[0] if focus_events else None
    if lead is not None and mode in ("peak", "now", "average") and measure["name"] not in lead["members"]:
        lead = None
    if story_mode and lead is not None:
        lines = list(lead["sentences"])
        short = " ".join(lines[:2])
        rest = focus_events[1:]
        comparable = [event for event in rest if event["score"] >= COMPARABLE_SHARE * lead["score"]]
        extra = []
        if comparable:
            extra.append("Also: " + comparable[0]["sentences"][0])
        elif rest:
            extra.append("Smaller: " + rest[0]["smaller"])
        summary = lines[:5 - 1 - len(extra)] + extra + [coverage_line(analysis, table)]
    elif story_mode:
        lines = quiet_sentences(analysis, measure)
        short = lines[0]
        summary = lines + [coverage_line(analysis, table)]
    else:
        short = lookup_answer(analysis, measure, mode)
        stats = reading_stats(analysis, measure)
        summary = [short, f"Typical {measure['label']} is {value_text(measure, median(measure['values']))} and it is {stats['direction']} now.",
                   coverage_line(analysis, table)]
        if lead is not None:
            summary.insert(2, lead["sentences"][0])
        summary = summary[:5]
    events_list = [dict(time=label, kind=kind, note=note) for label, kind, note in transcript_events(analysis, focus_events, measure, table)]
    result = dict(style=style, focus=measure["name"], readings=readings_text(analysis, table, lead, measure), table=None)
    if style == "short":
        result["answer"] = short
    elif style == "summary":
        result["answer"] = "\n".join(summary)
    elif style == "transcript":
        width = max(len(event["time"]) for event in events_list)
        result["answer"] = "\n".join(f"{event['time'].ljust(width)}  {event['kind'].upper().ljust(6)}  {event['note']}" for event in events_list)
    else:
        result["answer"] = short
        rows = event_rows(focus_events) or [["none", "none", "none", measure["label"], "none", value_text(measure, median(measure["values"])), "none", "none"]]
        result["table"] = dict(columns=["Start", "End", "Duration", "Signal", "Peak", "Normal", "Moved with it", "Top category"], rows=rows)
    result["events"] = events_list
    return result
