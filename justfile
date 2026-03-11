extract-poses scene:
    python parse_camera_poses.py {{scene}}.sav {{scene}}-poses.json
    python match_poses.py --scene {{scene}} {{scene}}-poses.json

reconstruct scene: (extract-poses scene)
    python reconstruct_3d.py --scene {{scene}} --all-views --poses {{scene}}-poses.json {{scene}}-mapping.json

