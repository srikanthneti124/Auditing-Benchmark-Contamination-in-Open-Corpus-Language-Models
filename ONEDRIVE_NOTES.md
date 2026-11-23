# OneDrive and this project

The project lives in `C:\Users\jahna\OneDrive\Pictures\Desktop\3`, inside the OneDrive-synced folder (checked 2026-10-29). Nothing has been moved or changed in OneDrive. The steps below are for you to do, if you choose.

## Why the location is risky
- **File locks.** While OneDrive uploads a file, it can briefly lock it. The overlap run saves `overlap_cache.json` every 200 new windows by writing `overlap_cache.json.tmp` and renaming it over the old file. A lock during that rename used to raise an error. Now (`atomicio.py`) the rename is retried 10 times, 0.2 s apart. If it still fails, the `.tmp` file is **kept** and a warning is written to `atomic_save_warnings.log`; the next run merges it. **Never delete a `*.tmp` file next to a cache.** Covered by `tests/test_storage_onedrive.py`; behaviour under a real OneDrive lock is UNVERIFIED.
- **Conflicted copies.** If the folder is open on two machines, or OneDrive syncs an old version back, you may get `overlap_cache-<PC name>.json` copies, or an older file may overwrite a newer one.
- **Files on demand.** If "Free up space" turns a file into an online-only placeholder, opening it triggers a download. A script reading it may stall or fail while offline.
- **Load and noise.** The cache (about 2.2 MB, rewritten often), `results.csv`, logs and `paper_outputs\` would be uploaded again and again during multi-hour runs.

## Pause sync during long runs
OneDrive icon in the taskbar → gear (Help & Settings) → **Pause syncing** → 24 hours. Resume the same way after `run_overlap.bat` / `run_gpu_stages.bat` finish. Pausing only stops syncing; it does not move or delete files.

## Move the project to a local folder safely (optional)
Do this only when **no** `overlap.py`, `evaluate.py` or `itemloglik.py` is running. Check with:
`powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Select ProcessId,CommandLine"`

1. Pause OneDrive (above). Make sure no file is online-only: right-click the folder → **Always keep on this device**, and wait for the green ticks.
2. Copy (do not move) to a local folder you choose, for example `C:\research\contam`:
   `robocopy "C:\Users\jahna\OneDrive\Pictures\Desktop\3" "C:\research\contam" /E /COPY:DAT /R:2 /W:2`
3. Verify the copy with hashes (prints nothing if every file matches):
   ```
   powershell -NoProfile -Command "$a='C:\Users\jahna\OneDrive\Pictures\Desktop\3'; $b='C:\research\contam'; Get-ChildItem $a -Recurse -File | ForEach-Object { $r=$_.FullName.Substring($a.Length); if ((Get-FileHash $_.FullName).Hash -ne (Get-FileHash ($b+$r)).Hash) { 'DIFFERENT: '+$r } }"
   ```
   At minimum, `overlap_cache.json` must have the same SHA-256 in both places: `Get-FileHash overlap_cache.json`.
4. In the new folder, run `python -m pytest tests -q` and then `python preflight_api.py`.
5. Use only the new folder from then on. Keep the OneDrive copy until a full run has succeeded from the new location; delete it later, yourself.

The model weights are not in the project folder (Hugging Face cache, see README.md "Storage"), so they are unaffected by OneDrive.
