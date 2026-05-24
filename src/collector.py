import pyalex
from pyalex import Works
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from collections import deque, defaultdict
from typing import List, Dict, Optional, Set, Any, Tuple
import re
import json
import csv
import time
import logging
from tqdm.auto import tqdm

from .models import WorkMetadata
from .configs import SeedScorerConfig, ExpansionConfig

class EpidemicOfScientificKnowledgeModel:
    def __init__(
        self,
        model: SentenceTransformer,
        bidirectional: bool = False,
        api_delay: float = 0.3
    ):
        self.model = model
        self.bidirectional = bidirectional
        self.api_delay = api_delay

        self.query: str = ""
        self._query_emb: Optional[np.ndarray] = None

        # Data
        self.seed_works: List[str] = []
        self.expanded_works: Dict[str, WorkMetadata] = {} # infective (broad=0)
        self.broad_works: Dict[str, WorkMetadata] = {}    # susceptible (broad=1)
        self.author_info: Dict[str, Dict[str, Any]] = {}
        self.visited_works: Set[str] = set()

        self._works_cache: Dict[str, Dict[str, Any]] = {}
        self._last_api_call: float = 0

        # logging
        self._initial_pool_data = {}

    # === LIMITS ===

    def _rate_limit_wait(self) -> None:
        """Ensure minimum delay between API calls"""
        elapsed = time.time() - self._last_api_call
        if elapsed < self.api_delay:
            time.sleep(self.api_delay - elapsed)
        self._last_api_call = time.time()

    # === OPENALEX API ===

    def _get_work(self, work_id: str) -> Dict[str, Any]:
        """Get work with caching to avoid duplicate API calls"""
        if work_id in self._works_cache:
            return self._works_cache[work_id]
        
        self._rate_limit_wait()
        if work_id.startswith("https://openalex.org/"):
            clean_id = work_id.split("/")[-1]
        else:
            clean_id = work_id
        work = Works()[clean_id]
        
        self._works_cache[work_id] = work
        return work


    # === EMBEDDINGS AND SCORES ===

    def _prepare_query(self, query_text: str) -> np.ndarray:
        """Calculate query embedding"""
        self.query = query_text.strip().lower()
        
        emb = self.model.encode(
            [self.query], 
            normalize_embeddings=True, 
            show_progress_bar=False
        )
        self._query_emb = emb[0]
        logging.info(f"Query embedding computed for: '{query_text}'")
        return self._query_emb


    def _prepare_text_for_embedding(self, work: WorkMetadata) -> str:
        """Prepare title and abstract for embedding"""
        text = f"{work.title} [SEP] {work.abstract}".strip()
        return text if text.replace("[SEP]", "").strip() else ""


    def _compute_semantic_similarity(self, text: str) -> float:
        """
        Compute similarity for query and article
        """        
        if not text.strip():
            return 0.0
        
        work_emb = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]
        
        sim = float(np.dot(self._query_emb, work_emb))
        return max(0.0, min(1.0, sim))


    def _compute_hybrid_scores(
        self,
        pool: List[WorkMetadata],
        config: SeedScorerConfig
    ) -> np.ndarray:
        """
        Calculate hybrid score.
        Update fields similarity_to_query and hybrid_score in-place.
        """
        # bathch variant
        texts: List[str] = []
        valid_indices: List[int] = []
        for i, work in enumerate(pool):
            text = self._prepare_text_for_embedding(work)
            if text:
                texts.append(text)
                valid_indices.append(i)
            else:
                pool[i].similarity_to_query = 0.0

        if not texts:
            raise ValueError("No valid texts found in pool for embedding")

        work_embs = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=config.batch_size,
            show_progress_bar=True
        )

        # hybrid part
        # similarities
        sims = work_embs @ self._query_emb
        sims = np.clip(sims, 0.0, 1.0)

        # citations
        citations = np.array([pool[i].cited_by_count for i in valid_indices])
        log_cits = np.log1p(citations)
        max_log_cit = np.max(log_cits) if log_cits.size > 0 else 1.0
        cit_sims = log_cits / max_log_cit if max_log_cit > 0 else np.zeros_like(log_cits)

        # meta (words intersection)
        query_tokens = set(re.findall(r'\b\w+\b', self.query.lower())) if self.query else set()
        meta_sims = np.zeros(len(valid_indices))
        if query_tokens:
            for idx, i in enumerate(valid_indices):
                work_text = f"{pool[i].title} {pool[i].abstract}".lower()
                work_tokens = set(re.findall(r'\b\w+\b', work_text))
                if work_tokens:
                    intersection = len(query_tokens & work_tokens)
                    union = len(query_tokens | work_tokens)
                    meta_sims[idx] = intersection / union if union > 0 else 0.0

        scores = np.zeros(len(pool))
        for idx, i in enumerate(valid_indices):
            scores[i] = (
                config.alpha_sem * sims[idx] +
                config.alpha_cit * cit_sims[idx] +
                config.alpha_meta * meta_sims[idx]
            )
            pool[i].similarity_to_query = float(sims[idx])

        return scores


    def _compute_expansion_thresholds(self, pool: List[WorkMetadata], config: SeedScorerConfig) -> Tuple[float, float]:
        
        sims = np.array([w.similarity_to_query for w in pool if w.similarity_to_query > 0])
        
        tau_infective = float(np.percentile(sims, config.expansion_infected_percentile * 100))
        tau_susceptible = float(np.percentile(sims, config.expansion_susceptible_percentile * 100))
        
        if tau_susceptible >= tau_infective:
            raise ValueError("Susceptible treshold is greater then infective")
            
        return tau_infective, tau_susceptible

    # === DATA ===

    def _extract_work_data(self, work_id: str) -> Optional[WorkMetadata]:

        try:
            if work_id in self._works_cache:
                work = self._works_cache[work_id]
            else:
                work = self._get_work(work_id)
            
            authorships = work.get("authorships", [])
            authors = [a["author"]["id"] for a in authorships if a.get("author") and a["author"].get("id")]
            author_names = [a["author"]["display_name"] for a in authorships if a.get("author") and a["author"].get("display_name")]
            
            primary_topic = None
            loc = work.get("primary_topic")
            if loc and isinstance(loc, dict):
                primary_topic = loc.get("display_name")
    
            abstract_dict = work.get("abstract_inverted_index", {})
            abstract = ""
            if isinstance(abstract_dict, dict):
                words = []
                for word, positions in abstract_dict.items():
                    if isinstance(positions, list):
                        for pos in positions:
                            if isinstance(pos, int):
                                words.append((pos, word))
                words.sort(key=lambda x: x[0])
                abstract = " ".join(word for _, word in words)
    
            keywords = work.get('keywords', [])
            topics = work.get('topics', [])
    
            return WorkMetadata(
                id=work.get("id", ""),
                title=work.get("title", ""),
                abstract=abstract,
                publication_year=work.get("publication_year"),
                cited_by_count=work.get("cited_by_count", 0),
                referenced_works=work.get("referenced_works", []),
                authors=authors,
                author_names=author_names,
                topics=[t.get("id", '') for t in topics if t.get("id")],
                topic_names=[t.get("display_name", '') for t in topics if t.get("display_name")],
                keywords=[kw.get("display_name", '') for kw in keywords if kw.get('display_name')],
                primary_topic=primary_topic,
                work_type=work.get("type", '')
            )
        except Exception as e:
            #logging.warning(f"Failed to extract metadata for {work_id}: {e}")
            return None


    def _process_new_work(
        self,
        work_id: str,
        depth: int,
        queue: deque,
        parent_id: Optional[str],
        config: ExpansionConfig
    ) -> None:

        if work_id in self.visited_works:
            return
        self.visited_works.add(work_id)

        try:
            work_data = self._extract_work_data(work_id)
            if work_data is None:
                #logging.warning(f"Data extraction failed for {work_id}. Skipping.")
                return

            if work_data.work_type not in {"article"}:
                return

            text = self._prepare_text_for_embedding(work_data)
            similarity = self._compute_semantic_similarity(text)
            
            work_data.similarity_to_query = similarity
            work_data.depth = depth

            if similarity >= config.tau_infective:
                work_data.state = "infective"
                self.expanded_works[work_id] = work_data
                queue.append((work_id, depth + 1, parent_id))
            elif similarity >= config.tau_susceptible:
                work_data.state = "susceptible"
                self.broad_works[work_id] = work_data

            ### logging
            if work_id in self._initial_pool_data:
                data = self._initial_pool_data[work_id]
                if not data["is_seed"]:
                    data["was_recovered"] = True
                    data["recovery_depth"] = int(depth)
                    data["similarity"] = float(similarity)
                    data["state"] = work_data.state
            ###

        except Exception as e:
            logging.warning(f"Error processing work {work_id} at depth {depth}: {e}")


    def _update_author_statistics(self) -> None:
        
        def _update_author_info(author_id: str, work_data: WorkMetadata, is_infective: bool) -> None:
            year = work_data.publication_year
            
            if author_id not in self.author_info:
                self.author_info[author_id] = {
                    "first_year": year,
                    "first_infective_year": None,
                    "infection_source_authors": None,
                    "infection_source_works": None,
                    "last_year": year,
                    "total_works": 0,
                    "infective_works": 0,
                    "susceptible_works": 0,
                    "topics": defaultdict(int),
                    "works_list": []
                }
            
            info = self.author_info[author_id]
            
            if year is not None:
                if info["first_year"] is None or year < info["first_year"]:
                    info["first_year"] = year
                if info["last_year"] is None or year > info["last_year"]:
                    info["last_year"] = year
            
            info["total_works"] += 1
            info["works_list"].append(work_data.id)
            
            if is_infective:
                info["infective_works"] += 1
                if info["first_infective_year"] is None or (year is not None and year < info["first_infective_year"]):
                    info["first_infective_year"] = year
                    cited_infective = [wid for wid in work_data.referenced_works if wid in self.expanded_works]
                    if cited_infective:
                        source_authors = set()
                        for wid in cited_infective:
                            src_work = self.expanded_works[wid]
                            source_authors.update(src_work.authors)
                        info["infection_source_authors"] = list(source_authors)
                        info["infection_source_works"] = cited_infective
            else:
                info["susceptible_works"] += 1
            
            if work_data.primary_topic:
                info["topics"][work_data.primary_topic] += 1

        # updating
        with tqdm(total=(len(self.expanded_works) + len(self.broad_works)), mininterval=1.0, desc="Authors data update") as pbar:
            for work_data in self.expanded_works.values():
                for aid in work_data.authors:
                    _update_author_info(aid, work_data, is_infective=True)
            
            for work_data in self.broad_works.values():
                for aid in work_data.authors:
                    _update_author_info(aid, work_data, is_infective=False)
                
    
    def _fetch_candidate_pool(
        self,
        queries: List[str],
        year_range: Tuple[int, int],
        max_candidates_by_query: int = 5000
    ) -> List[WorkMetadata]:
        
        if not queries:
            raise ValueError("There is found no query")
        if year_range[0] > year_range[1]:
            raise ValueError("Unvalid years")

        pool: List[WorkMetadata] = []
        seen_ids: Set[str] = set()
        year_filter = f"{year_range[0]}-{year_range[1]}"

        for num, query in enumerate(queries):
            logging.info(f"Fetching candidates for query: '{query}'")
            try:
                search = Works().search(query).filter(
                    publication_year=year_filter,
                    type="|".join(["article"]),
                    has_abstract="true"
                )

                works_collected = 0
                cursor = "*"

                with tqdm(total=max_candidates_by_query, desc=f'{query}') as pbar:
                
                    while works_collected < max_candidates_by_query:
                        #self._rate_limit_wait()
                        results = search.get(per_page=200, cursor=cursor)
                        meta = results.meta
                        works_page = list(results)
                        
                        if not works_page:
                            break
    
                        for work in works_page:
                            if works_collected >= max_candidates_by_query:
                                break
    
                            work_id = work.get("id")
                            if not work_id or work_id in seen_ids:
                                continue
    
                            seen_ids.add(work_id)
                            self._works_cache[work_id] = work
    
                            work_data = self._extract_work_data(work_id)
                            if work_data is None:
                                continue
                            pool.append(work_data)
                            
                            works_collected += 1
                            pbar.update(1)

                        cursor = meta.get('next_cursor')
                        if not cursor:
                            break
                logging.info(f'Get {works_collected} via query "{query}"')

            except Exception as e:
                logging.error(f"Failed to process query '{query}': {e}")

        logging.info(f"Retrieved {len(pool)} unique candidates from {len(seen_ids)} hits.")
        return pool

    # === PUBLIC API ===

    def get_seed_works_by_query(
        self,
        query: str,
        queries_for_search: Optional[List[str]] = None,
        config: Optional[SeedScorerConfig] = None,
        max_candidates_by_query: int = 5000,
        year_range: Tuple[int, int] = (2018, 2024),
    ) -> Optional[Tuple[float, float]]:
        if not queries_for_search:
            queries_for_search = [query]
        pool = self._fetch_candidate_pool(queries_for_search, year_range, max_candidates_by_query)

        if not pool:
            logging.warning("Candidate pool is empty. No seeds can be selected.")
            return None

        self._prepare_query(query)
        if self._query_emb is None:
            raise RuntimeError("Failed to compute query embedding")

        if config is None:
            config = SeedScorerConfig()

        scores = self._compute_hybrid_scores(pool, config)
        threshold = float(np.percentile(scores, config.seed_percentile * 100))
        print(f"threshold {threshold}")

        ### logging
        for work, score in zip(pool, scores):
            self._initial_pool_data[work.id] = {
                "score": float(score),
                "is_seed": bool(score >= threshold),
                "was_recovered": False,
                "recovery_depth": -1,
                "similarity": 0,
                "state": "NaN"
            }
        ###

        self.seed_works = []
        for work, score in zip(pool, scores):
            if score >= threshold:
                work.is_seed = True
                self.seed_works.append(work.id)

        tau_infective, tau_susceptible = self._compute_expansion_thresholds(pool, config)

        tau_infective = min(tau_infective, 0.85)
        tau_susceptible = min(tau_susceptible, 0.6)

        return tau_infective, tau_susceptible


    def semantic_expansion_bfs(self, config: Optional[ExpansionConfig] = None) -> None:
        if config is None:
            config = ExpansionConfig()
            
        if not self.seed_works:
            raise ValueError("No seed works. Run get_seed_works_by_query first.")

        queue: deque[Tuple[str, int, Optional[str]]] = deque()

        for wid in self.seed_works:
            self.visited_works.add(wid)
            work_data = self._extract_work_data(wid)
            if work_data is None:
                logging.warning(f"Seed work {wid} metadata extraction failed. Skipping seed.")
                continue
                
            work_data.is_seed = True
            work_data.state = "infective"
            work_data.depth = 0
            
            text = self._prepare_text_for_embedding(work_data)
            work_data.similarity_to_query = self._compute_semantic_similarity(text)
            
            self.expanded_works[wid] = work_data
            queue.append((wid, 0, None))

        with tqdm(total=config.max_works, desc="BFS Expansion", mininterval=1.0) as pbar:
            pbar.update(len(self.expanded_works) + len(self.broad_works))
            
            while queue and (len(self.expanded_works) + len(self.broad_works)) < config.max_works:
                count_before = len(self.expanded_works) + len(self.broad_works)
                current_id, depth, parent_id = queue.popleft()
                
                if depth >= config.max_depth:
                    continue
    
                try:
                    raw_work = self._get_work(current_id)
    
                    # FORWARD:
                    if self.bidirectional:
                        # self._rate_limit_wait()
                        citing_list = Works().filter(cites=current_id).get(per_page=config.max_cit_per_work)
                        for citing in citing_list:
                            cid = citing.get("id")
                            if cid and cid not in self.visited_works:
                                self._process_new_work(cid, depth + 1, queue, parent_id=current_id, config=config)
    
                    # BACKWARD:
                    refs = raw_work.get("referenced_works", [])
                    for ref_id in refs[:config.max_refs_per_work]:
                        if ref_id and ref_id not in self.visited_works:
                            self._process_new_work(ref_id, depth + 1, queue, parent_id=current_id, config=config)
                                
                except Exception as e:
                    logging.warning(f"Error during BFS expansion for work {current_id}: {e}")
    
                pbar.update(len(self.expanded_works) + len(self.broad_works) - count_before)
                time.sleep(0.1)

        logging.info(
            f"BFS complete: Infective={len(self.expanded_works)}, "
            f"Susceptible={len(self.broad_works)}, Visited={len(self.visited_works)}"
        )

        self._update_author_statistics()

        ### logging
        output = {
            "query": self.query,
            "total_pool_size": len(self._initial_pool_data),
            "data": self._initial_pool_data
        }
        with open(f"{self.query} seed_stats", 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        ###


    def save_results(self, field_name: str = "scientific_field") -> None:
    
        def safe_join(lst: List[Any], sep: str = ';') -> str:
            if not lst:
                return ""
            return sep.join(str(x) for x in lst if x is not None)
        
        # === works.csv ===
        all_works: List[Dict[str, Any]] = []
        
        for wid, work_meta in self.expanded_works.items():
            all_works.append({
                'id': work_meta.id,
                'title': work_meta.title,
                'publication_year': work_meta.publication_year,
                'authors': work_meta.authors,
                'author_names': work_meta.author_names,
                'cited_by_count': work_meta.cited_by_count,
                'referenced_works': work_meta.referenced_works,
                'abstract': work_meta.abstract,
                'keywords': work_meta.keywords,
                'topics': work_meta.topics,
                'topic_names': work_meta.topic_names,
                'primary_topic': work_meta.primary_topic,
                'broad': 0,  # infective
                'similarity': work_meta.similarity_to_query,
                'depth': work_meta.depth,
                'state': work_meta.state,
                'is_seed': work_meta.is_seed
            })
        
        for wid, work_meta in self.broad_works.items():
            all_works.append({
                'id': work_meta.id,
                'title': work_meta.title,
                'publication_year': work_meta.publication_year,
                'authors': work_meta.authors,
                'author_names': work_meta.author_names,
                'cited_by_count': work_meta.cited_by_count,
                'referenced_works': work_meta.referenced_works,
                'abstract': work_meta.abstract,
                'keywords': work_meta.keywords,
                'topics': work_meta.topics,
                'topic_names': work_meta.topic_names,
                'primary_topic': work_meta.primary_topic,
                'broad': 1,  # susceptible
                'similarity': work_meta.similarity_to_query,
                'depth': work_meta.depth,
                'state': work_meta.state,
                'is_seed': work_meta.is_seed
            })
        
        if not all_works:
            logging.warning("No works to save")
            return
        
        fieldnames = [
            'id', 'title', 'publication_year', 'authors', 'author_names',
            'cited_by_count', 'referenced_works', 'abstract',
            'keywords', 'topics', 'topic_names', 'primary_topic',
            'broad', 'similarity', 'depth', 'state', 'is_seed'
        ]
        
        works_filename = f"{field_name}_works.csv"
        with open(works_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for work in all_works:
                csv_work = {
                    'id': work['id'],
                    'title': work['title'],
                    'publication_year': work['publication_year'] if work['publication_year'] is not None else '',
                    'authors': safe_join(work['authors']),
                    'author_names': safe_join(work['author_names']),
                    'cited_by_count': work['cited_by_count'],
                    'referenced_works': safe_join(work['referenced_works']),
                    'abstract': work['abstract'],
                    'keywords': safe_join(work['keywords']),
                    'topics': safe_join(work['topics']),
                    'topic_names': safe_join(work['topic_names']),
                    'primary_topic': work['primary_topic'] if work['primary_topic'] is not None else '',
                    'broad': work['broad'],
                    'similarity': work['similarity'],
                    'depth': work['depth'],
                    'state': work['state'],
                    'is_seed': work['is_seed']
                }
                writer.writerow(csv_work)
        
        logging.info(f"Saved {len(all_works)} works in {works_filename}")
        
        # === authors.csv ===
        if not self.author_info:
            logging.info("There is no author's info to save")
            return
        
        fieldnames_authors = [
            'author_id', 'first_publication_year',
            'first_infective_year', 'infection_source_authors',
            'infection_source_works', 'last_publication_year',
            'total_works', 'infective_works', 'susceptible_works',
            'is_infective', 'is_susceptible', 'works_list', 'main_topic'
        ]
        
        authors_filename = f"{field_name}_authors.csv"
        with open(authors_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames_authors)
            writer.writeheader()
            
            for author_id, info in self.author_info.items():
                is_infective = info['infective_works'] > 0
                is_susceptible = info['susceptible_works'] > 0 and not is_infective
                
                main_topic = max(info['topics'], key=info['topics'].get) if info['topics'] else None
                
                infection_source_authors = safe_join(info.get('infection_source_authors', []))
                infection_source_works = safe_join(info.get('infection_source_works', []))
                
                author_data = {
                    'author_id': author_id if author_id is not None else '',
                    'first_publication_year': info.get('first_year', ''),
                    'first_infective_year': info.get('first_infective_year', ''),
                    'infection_source_authors': infection_source_authors,
                    'infection_source_works': infection_source_works,
                    'last_publication_year': info.get('last_year', ''),
                    'total_works': info.get('total_works', 0),
                    'infective_works': info.get('infective_works', 0),
                    'susceptible_works': info.get('susceptible_works', 0),
                    'is_infective': is_infective,
                    'is_susceptible': is_susceptible,
                    'works_list': safe_join(info.get('works_list', [])),
                    'main_topic': main_topic
                }
                writer.writerow(author_data)
                
        logging.info(f"Saved {len(self.author_info)} authors in {authors_filename}")
        
        # === citations.csv ===
        work_id_to_year = {w['id']: w['publication_year'] for w in all_works}
        work_id_to_broad = {w['id']: w['broad'] for w in all_works}
        work_ids_set = set(work_id_to_year.keys())
        
        citations: List[Dict[str, Any]] = []
        for w in all_works:
            citing_id = w['id']
            citing_year = w['publication_year']
            citing_broad = work_id_to_broad[citing_id]
            
            ref_list = w.get('referenced_works', [])
            if isinstance(ref_list, str):
                ref_ids = ref_list.split(';') if ref_list else []
            else:
                ref_ids = ref_list
            
            for cited_id in ref_ids:
                if cited_id in work_ids_set:
                    cited_year = work_id_to_year[cited_id]
                    cited_broad = work_id_to_broad[cited_id]

                    if (citing_year is not None and cited_year is not None 
                        and citing_year >= cited_year):
                        citations.append({
                            'citing_work_id': citing_id,
                            'cited_work_id': cited_id,
                            'citing_year': citing_year,
                            'cited_year': cited_year,
                            'citing_broad': citing_broad,
                            'cited_broad': cited_broad
                        })
        
        if citations:
            citations_filename = f"{field_name}_citations.csv"
            citations_df = pd.DataFrame(citations)
            citations_df.to_csv(citations_filename, index=False)
            logging.info(f"Saved {len(citations)} citation edges in {citations_filename}")
