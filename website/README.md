# PYNQ Process website

The public site where people download the Mac app. One static page, one stylesheet, one script, no build tools. Every "Get the app" button downloads the zip directly.

## Run it

```sh
sh website/sync-download.sh
python3 -m http.server 5330 -d website
```

`sync-download.sh` copies `build/mac-kit/pinkProcess-Mac-arm64.zip` into `website/downloads/`, which is what the Download button links to. The repo is private, so the zip has to ship with the site instead of linking to GitHub.

## Deploy

Point a Vercel project at the `website` folder. `vercel.json` runs the sync script on build, so every deploy serves the zip from the same commit. Screenshots in `assets/shots/` come from the app running against the simulated board (`PYNQ_SIM_BOARD=1`); retake them when the interface changes.
