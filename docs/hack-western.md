# Hack Western 13 plan

Source: the HW13 Devpost page and hacker package, read 2026-11-21.

## Dates

- Hacking ends and Devpost closes Sunday Nov 22, 9:00 a.m.
- Sponsor-track judging from 10:00 a.m., second floor.
- Demo fair and general judging 11:30 a.m., Grand Hall. Three judges, each gets 3 minutes of demo and 2 minutes of questions. Live demo, no slides.

## Rule to settle first

"Build something new during the hacking period. Projects started before the event can't be submitted." This repo has commits from Nov 18 and 19, and `research.md` reports work from Nov 4 onward. Ask an organizer how to present that before submitting. The cartridge system, kernel, decks and app were all written during the event.

## Tracks

| track | fit | what to show |
|---|---|---|
| First, second, third overall | main target | the full demo below |
| Best Hardware Prototype by PCBWay | strong | the board, the kernel, and the next board: a cartridge slot carrier with SPI flash per cartridge |
| Best Health Hack by Sun Life | strong | First aid cartridge. Who it helps: untrained staff in the first five minutes |
| Canada Life Future Me Challenge | strong | Future me cartridge: TFSA, RRSP, FHSA, matching, starting small |
| Organizers' Choice: Most Adventurous | good | Backcountry cartridge and the theme "discover the unknown" |
| MLH Best Use of Gemini API | possible | use Gemini as the teacher that drafts decks and questions; the device stays offline |
| Best Use of QNX | no | needs QNX on a Raspberry Pi 5 with a QNX AI module |
| TinyFish, Solana, Tiger Data, Auth0, Vultr | no | unrelated |

## Judging criteria and where we score

| criterion | our answer |
|---|---|
| Innovation and creativity | expertise as a swappable 64 KB memory image on a fixed circuit |
| Technical complexity | custom kernel, fixed-point replay, bit-exact test vectors, weight-reading research |
| Completeness | every number on screen is measured; say plainly what runs on the board and what is software replay |
| Design | one input, numbered steps, sheet tabs for fields |

## Three-minute demo

1. (20 s) Hand the judge the cable. They open the browser. No install, no network.
2. (40 s) They type an emergency in their own words. Steps appear. Point at the decision time.
3. (40 s) They type a money question. The tab changes by itself. Say: the circuit did not change, 64 KB of memory did.
4. (30 s) They ask for Python code or farming advice. It refuses.
5. (30 s) Show the testbench line: 3,264 scores, 0 mismatches. The device does what the file says.
6. (20 s) Close: every job that needs an expert for five minutes can have one for $150.

Questions to be ready for: why not a phone app, how a new field gets made, what happens when it is wrong, how big a model the board can hold.
