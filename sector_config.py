# sector_config.py — AI 供應鏈產業分群設定
# 配合 day27_ranking_3day.py 使用，新增產業輪動特徵

SECTOR_MAP = {
    # ── A：晶片設計 ──────────────────────────────
    "聯發科":     "A_chip_design",
    "世芯-KY":    "A_chip_design",
    "訊芯-KY":    "A_chip_design",
    "晶心科":     "A_chip_design",
    "智原":       "A_chip_design",
    "M31":       "A_chip_design",
    "聯詠":       "A_chip_design",
    "原相":       "A_chip_design",
    "光寶科":     "A_chip_design",
    "所羅門":     "A_chip_design",

    # ── B：記憶體 ─────────────────────────────────
    "南亞科":     "B_memory",
    "華邦電":     "B_memory",
    "旺宏":       "B_memory",

    # ── C：半導體製造 / 封測 ───────────────────────
    "台積電":     "C_foundry_osat",
    "日月光投控":  "C_foundry_osat",
    "力成":       "C_foundry_osat",
    "穩懋":       "C_foundry_osat",
    "超豐":       "C_foundry_osat",
    "IET-KY":    "C_foundry_osat",
    "台表科":     "C_foundry_osat",
    "全新":       "C_foundry_osat",

    # ── D：PCB / 載板 ────────────────────────────
    "南電":       "D_pcb",
    "欣興":       "D_pcb",
    "臻鼎-KY":    "D_pcb",
    "家登":       "D_pcb",
    "弘塑":       "D_pcb",
    "台表科":     "D_pcb",

    # ── E：AI 伺服器 / 系統組裝 ──────────────────
    "廣達":       "E_server",
    "英業達":     "E_server",
    "緯創":       "E_server",
    "鴻海":       "E_server",
    "緯穎":       "E_server",
    "樺漢":       "E_server",
    "研華":       "E_server",

    # ── F：散熱 / 電源 / 連接器 ──────────────────
    "奇鋐":       "F_thermal_power",
    "貿聯-KY":    "F_thermal_power",
    "台達電子工業": "F_thermal_power",
    "群光電子":    "F_thermal_power",

    # ── G：被動元件 ──────────────────────────────
    "國巨":       "G_passive",
    "凱美":       "G_passive",
    "尼克森":     "G_passive",
    "禾伸堂企業":  "G_passive",

    # ── H：半導體材料 / 設備 ──────────────────────
    "志聖":       "H_semi_equip",
    "中砂":       "H_semi_equip",
    "家碩":       "H_semi_equip",
    "商丞":       "H_semi_equip",
    "鈦昇":       "H_semi_equip",
    "意德士":     "H_semi_equip",

    # ── I：其他電子 / 主機板 ─────────────────────
    "技嘉":       "I_other_elec",
    "華碩":       "I_other_elec",
    "群創":       "I_other_elec",
    "宏達電":     "I_other_elec",
    "晟銘電":     "I_other_elec",
    "倉佑":       "I_other_elec",
    "東陽":       "I_other_elec",
    "文曄":       "I_other_elec",
    "慧洋-KY":    "I_other_elec",
}

SECTOR_NAMES = {
    "A_chip_design":   "晶片設計",
    "B_memory":        "記憶體",
    "C_foundry_osat":  "製造封測",
    "D_pcb":           "PCB載板",
    "E_server":        "AI伺服器",
    "F_thermal_power": "散熱電源",
    "G_passive":       "被動元件",
    "H_semi_equip":    "半導體設備",
    "I_other_elec":    "其他電子",
}
