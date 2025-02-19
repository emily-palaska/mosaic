from docarray import BaseDoc
from docarray.typing import NdArray

from .config import FEATURE_SIZE

class BlockMergerDoc(BaseDoc):
    label: str = ''
    block: list = []
    embedding: NdArray[FEATURE_SIZE]