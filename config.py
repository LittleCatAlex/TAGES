#config.py
"""
TAGES 系統全域地質設定檔 (Global Geological Configuration)
未來如果要新增岩石種類、修改顏色、調整物理波速，只要修改這裡即可！
"""

LITHOLOGY_DICT = {
    0: {"name": "Unknown",       "vp": 1500, "vs": 500,  "color": "lightgray",     "keywords": []},
    1: {"name": "Topsoil",       "vp": 800,  "vs": 300,  "color": "khaki",         "keywords": ["表土", "回填", "雜土", "瀝青", "混凝土", "植生"]},
    2: {"name": "Mud / Clay",    "vp": 1600, "vs": 600,  "color": "saddlebrown",   "keywords": ["泥", "頁", "黏土", "壤土"]},
    3: {"name": "Silt",          "vp": 2000, "vs": 800,  "color": "tan",           "keywords": ["粉土", "粉砂"]},
    4: {"name": "Sandstone",     "vp": 2800, "vs": 1400, "color": "sandybrown",    "keywords": ["砂"]},
    5: {"name": "Gravel",        "vp": 2400, "vs": 1200, "color": "dimgray",       "keywords": ["礫", "卵石", "塊石", "圓礫"]},
    6: {"name": "Interbedded",   "vp": 2200, "vs": 1000, "color": "olivedrab",     "keywords": ["互層"]},
    7: {"name": "Hard Bedrock",  "vp": 5000, "vs": 2800, "color": "darkslategray", "keywords": ["變質", "板岩", "片岩", "大理岩", "安山岩", "玄武岩", "岩盤"]}
}

# NLP 判定優先順序
# 數字代表上面的 ID，越前面的會越先檢查
NLP_PRIORITY = [1, 6, 7, 5, 3, 4, 2]