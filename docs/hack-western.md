# Hack Western 13 plan

Source: the HW13 Devpost page, its rules tab and the hacker package, read 2026-11-21.

## Dates

- Hacking ends and Devpost closes Sunday Nov 22, 9:00 a.m. Submissions can't be edited afterwards.
- Sponsor-track judging from 10:00 a.m., second floor, for shortlisted teams only.
- Demo fair and general judging in the Grand Hall. The hacker package says 11:30 a.m.; the Devpost rules say 12:00 to 1:00 p.m. Check the live site.
- Three judges, 3 minutes of live demo each, then 1 to 2 minutes of questions. No slides.

## Entering several tracks

One project per team, and the rules set no limit on tracks. Select each sponsor track on **both** the live site and the Devpost submission; they must match. Sponsors shortlist from submissions, so the Devpost write-up has to make each track's case.

We enter five:

| track | prize | entered how |
|---|---|---|
| Overall (first, second, third) | iPad, AirPods Pro, Fitbit per member | general judging |
| Organizers' Choice: Most Adventurous | Arc'teryx toque per member | general judging |
| Best Health Hack by Sun Life | $100 Amazon card per member | sponsor track |
| The Canada Life Future Me Challenge | $100 Visa card per member | sponsor track |
| Best Hardware Prototype by PCBWay | US$50 PCBWay credit | sponsor track |

Not entering: QNX (needs QNX OS on their Raspberry Pi), Gemini, TinyFish, Solana, Tiger Data, Auth0, Vultr, Entire, GoDaddy.

## Overall

Criteria from Devpost, and what to show for each:

| criterion | show |
|---|---|
| Innovation and creativity | expertise as a swappable 64 KB memory image on a fixed circuit |
| Technical complexity | custom kernel, fixed-point arithmetic, 3,264 test scores with 0 mismatches, the pruning study behind the design |
| Completeness | it answers, switches fields and refuses, live. Say which parts run on the board |
| Design | one input, numbered steps, sheet tabs for fields |

Three-minute demo:

1. (20 s) Hand the judge the cable. They open a browser. No install, no network.
2. (40 s) They type an emergency in their own words. Steps appear.
3. (40 s) They type a money question. The tab changes by itself. "The circuit did not change. 64 KB of memory did."
4. (30 s) They ask for Python code or farming advice. It refuses.
5. (30 s) Show the testbench line: 3,264 scores, 0 mismatches.
6. (20 s) "Every job that needs an expert for five minutes can have one for $150."

Be ready for: why not a phone app, how a new field gets made, what happens when it is wrong, how big a model the board can hold.

## Best Health Hack by Sun Life

Brief: "Build a creative technical solution to a real health problem. In your pitch, show who it helps and what impact it could have."

- **Cartridge:** First aid. Ten cards: bleeding, choking, burns, stroke, CPR, low blood sugar, severe allergy, heat, nosebleed, seizure.
- **Who it helps:** the untrained person who is first on the scene: a warehouse worker, a camp counsellor, a new care aide.
- **Impact:** the right steps in the first minutes, with no signal and no account, and the same answer every time. Each card ends with when to call 911.
- **Demo:** judge describes an emergency in their own words; then ask it something medical it has no card for and show it refuse instead of guessing.
- **Say plainly:** the cards have not been reviewed by a clinician yet.

## The Canada Life Future Me Challenge

Brief: help Gen Z "take meaningful action toward their financial future through motivation, visualization, behavioral nudges, education, or innovation".

- **Cartridge:** Future me. Ten cards: starting small, TFSA or RRSP, FHSA, employer matching, emergency fund, compounding, debt or invest, budgeting, index funds, credit score.
- **Action:** every card is numbered steps that start with something doable today, such as a $25 automatic transfer.
- **Motivation:** the compounding card gives the two numbers that matter: $200 a month from 22 grows to about $485,000 by 65, from 32 about $250,000.
- **Why a device:** money advice a 20-year-old can trust has to be the same for everyone and free of upsell. The answers are fixed and anyone can read the file.
- **Demo:** "I only have 50 dollars a month, is it worth investing?"

## Best Hardware Prototype by PCBWay

Brief: "Build a hardware prototype that solves a real problem. Show how it works and what you'd fabricate for the next version."

- **Prototype:** PYNQ-Z2 running the cartridge app, plus the kernel in `hw/`.
- **How it works:** one fixed circuit of 16 accumulators; a cartridge is the contents of one block RAM.
- **Next version to fabricate:** a carrier board with a physical cartridge slot. Each cartridge is a small PCB with an SPI flash chip holding its 64 KB image and an edge connector. Push in a different cartridge and the device becomes a different expert. Add a small screen and three buttons and the laptop is no longer needed.

## Organizers' Choice: Most Adventurous

Brief: "a hack that enables others to discover the unknown."

- **Cartridge:** Backcountry. Ten cards: lost, hypothermia, water, bears, lightning, a night out, signalling, ticks, frostbite, campfires.
- **Line:** it lets someone walk into a field they know nothing about, a trail or a first job or a first paycheque, and act like they have done it before.
