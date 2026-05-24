from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class WorkMetadata:
    id: str
    title: str
    abstract: str
    publication_year: Optional[int]
    cited_by_count: int
    referenced_works: List[str]
    authors: List[str]
    author_names: List[str]
    topics: List[str]
    topic_names: List[str]
    keywords: List[str]
    primary_topic: Optional[str]
    
    similarity_to_query: float = 0.0
    state: str = "NaN"
    depth: int = -1
    is_seed: bool = False

    work_type: Optional[str] = None
