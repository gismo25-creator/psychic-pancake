# Remove Profile Manager (Governance) + make ACTIVE usable in Live (simple profiles)

This patch does two things:

1) Migrates `data/profiles/active.json` into the **simple Profile Library** so Live → Profiles (simple) can see it.
2) Removes/disables the "Profile Manager (Governance)" page from navigation and renames the page file if found.

## Apply

Unzip into your project root (same folder as `streamlit_app.py`), then run:

    python tools/simplify_profiles_remove_governance.py

Backups:
- `streamlit_app.py.bak`
- `data/profile_library/index.json.bak` (if it existed)

Then restart:
- `streamlit run streamlit_app.py`

## Result

In Live → Profiles (simple) you should see an entry like:
`ACTIVE_MIGRATED::<SYMBOL>::<TIMEFRAME>` and you can apply it directly.