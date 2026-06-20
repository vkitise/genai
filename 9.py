!pip install pydantic
!pip install wikipedia
from pydantic import BaseModel
import wikipedia
class InstitutionInfo(BaseModel):
    name:str
    summary:str
name = input("Enter Institution Name: ")
try:
    page = wikipedia.page(name)
    info = InstitutionInfo(
        name=name,
        summary=" ".join(page.summary.split(".")[:2])
    )
    print(info.model_dump_json(indent=4))
except:
    print("No information found")
