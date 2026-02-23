import uuid
from datetime import datetime
def new_id(): return uuid.uuid4().hex
def iso(dt: datetime): return dt.replace(microsecond=0).isoformat()+"Z"
