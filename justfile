extract-poses scene:
    uv run parse-camera-poses {{scene}}.sav {{scene}}-poses.json
    uv run match-poses --scene {{scene}} {{scene}}-poses.json

reconstruct scene: (extract-poses scene)
    uv run reconstruct-3d --scene {{scene}} --all-views --poses {{scene}}-poses.json {{scene}}-mapping.json
