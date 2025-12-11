# config.py
from pathlib import Path
import os

# ==== 0. 本地项目根目录（代码所在） ====
PROJECT_ROOT = Path(__file__).resolve().parent

# ==== 1. 外部硬盘项目根 ====
# 默认指向你现在的数据目录
DEFAULT_EXT_ROOT = Path("/Volumes") / "LYY_T7" / "13_抽象实验数据分析"
EXT_ROOT = Path(os.environ.get("ABSTRACT_EXP_ROOT", DEFAULT_EXT_ROOT))

# ==== 2. 顶层目录 ====
RAW_DIR   = EXT_ROOT / "1_rawdata"
DERIV_DIR = EXT_ROOT / "3_derivatives"
DOCS_DIR  = EXT_ROOT / "4_docs"
RESULT_DIR= EXT_ROOT / "5_result"

# ==== 3. 1_rawdata 子目录 ====
RAW_BEHAVIOR      = RAW_DIR / "behavior"
RAW_EVENT         = RAW_DIR / "event"
RAW_EYE           = RAW_DIR / "eyetracking"
RAW_PHYSIO        = RAW_DIR / "physio"
RAW_QUESTIONNAIRE = RAW_DIR / "questionnaire"

# ==== 4. 3_derivatives 子目录 ====
DERIV_BEHAVIOR    = DERIV_DIR / "behavior"
DERIV_INTEGRATION = DERIV_DIR / "integration"
DERIV_PHYSIO      = DERIV_DIR / "physio"
DERIV_EYE         = DERIV_DIR / "eyetracking"  # 眼动 summary 建议新建这个目录

# ==== 5. 5_result 子目录（按模态分一个层） ====
RES_BEHAVIOR = RESULT_DIR / "behavior"
RES_EYE      = RESULT_DIR / "eye"
RES_PHYSIO   = RESULT_DIR / "physio"

# ==== 6. 小工具函数，尽量都用这些，不要手写路径 ====
def raw_behavior(*parts) -> Path:
    return RAW_BEHAVIOR.joinpath(*parts)

def raw_event(*parts) -> Path:
    return RAW_EVENT.joinpath(*parts)

def raw_eye(*parts) -> Path:
    return RAW_EYE.joinpath(*parts)

def raw_physio(*parts) -> Path:
    return RAW_PHYSIO.joinpath(*parts)

def raw_questionnaire(*parts) -> Path:
    return RAW_QUESTIONNAIRE.joinpath(*parts)

def deriv_behavior(*parts) -> Path:
    return DERIV_BEHAVIOR.joinpath(*parts)

def deriv_integration(*parts) -> Path:
    return DERIV_INTEGRATION.joinpath(*parts)

def deriv_physio(*parts) -> Path:
    return DERIV_PHYSIO.joinpath(*parts)

def deriv_eye(*parts) -> Path:
    return DERIV_EYE.joinpath(*parts)

def res_behavior(*parts) -> Path:
    return RES_BEHAVIOR.joinpath(*parts)

def res_eye(*parts) -> Path:
    return RES_EYE.joinpath(*parts)

def res_physio(*parts) -> Path:
    return RES_PHYSIO.joinpath(*parts)

def result_path(*parts) -> Path:
    """通用：5_result 下的路径"""
    return RESULT_DIR.joinpath(*parts)
