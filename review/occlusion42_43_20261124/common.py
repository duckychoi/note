import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/"risk42_43_20261124"
sys.path.insert(0,str(OLD))
STORE=ROOT.parents[2]/"research-sources"/"occlusion42_mobile"
SEEDS=(6211,6221,6229,6247,6257)
ARMS=("legacy","plain_depth","plain_order","lines_depth","lines_order")
FAMILIES=dict(artroom=2,bandsaw=2,chess=3,curule=3,ladder=2,octogons=2,
              pendulum=2,podium=1,skates=2,skiboots=3,traproom=2)
ALL=tuple(f"{g}{i+1}" for g,n in FAMILIES.items() for i in range(n))
DEV=("artroom1","skiboots1")
TEST=tuple(s for s in ALL if s[:-1] not in ("artroom","skiboots"))


def save(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=1),encoding="utf8")
