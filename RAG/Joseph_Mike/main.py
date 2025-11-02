
"""
Knowledge Graph RAG Proof of Concept
Uses sentence-transformers for encoding and simple template generation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json


# Knowledge Graph Structure
@dataclass
class Node:
    id: str
    type: str
    properties: Dict[str, Any]
    
    def to_text(self) -> str:
        """Convert node to text representation for embedding."""
        parts = [f"type: {self.type}"]
        for key, value in self.properties.items():
            parts.append(f"{key}: {value}")
        return " | ".join(parts)


@dataclass
class Edge:
    source: str
    target: str
    relation: str


class KnowledgeGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
    
    def add_node(self, node: Node):
        self.nodes[node.id] = node
    
    def add_edge(self, edge: Edge):
        self.edges.append(edge)
    
    def get_node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)
    
    def get_connected_nodes(self, node_id: str, depth: int = 1) -> List[Node]:
        """Get nodes connected to given node within depth hops."""
        if depth == 0:
            return []
        
        connected_ids = set()
        to_explore = {node_id}
        explored = set()
        
        for _ in range(depth):
            current_level = set()
            for nid in to_explore:
                if nid in explored:
                    continue
                explored.add(nid)
                
                for edge in self.edges:
                    if edge.source == nid:
                        connected_ids.add(edge.target)
                        current_level.add(edge.target)
                    elif edge.target == nid:
                        connected_ids.add(edge.source)
                        current_level.add(edge.source)
            
            to_explore = current_level
        
        return [self.nodes[nid] for nid in connected_ids if nid in self.nodes]


class SimpleEncoder:
    """
    Simple bag-of-words encoder as baseline.
    """
    
    def __init__(self):
        self.vocab = {}
        self.embedding_dim = 100
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embedding vectors."""
        embeddings = []
        
        for text in texts:
            # Simple word frequency embedding
            words = text.lower().split()
            freq = {}
            for word in words:
                freq[word] = freq.get(word, 0) + 1
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
            
            # Create sparse vector
            vec = np.zeros(self.embedding_dim)
            for word, count in freq.items():
                idx = self.vocab[word] % self.embedding_dim
                vec[idx] += count
            
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            
            embeddings.append(vec)
        
        return np.array(embeddings)


class SentenceTransformerEncoder:
    """
    Wrapper for sentence-transformers with optimization.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Loaded {model_name} with dimension {self.embedding_dim}")
            
        except ImportError:
            print("Install with: pip install sentence-transformers")
            raise
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embedding vectors."""
        # convert_to_numpy=True ensures we get numpy arrays
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings


class KnowledgeGraphRAG:
    def __init__(self, kg: KnowledgeGraph, encoder=None):
        self.kg = kg
        self.encoder = encoder or SimpleEncoder()
        
        # Pre-compute embeddings for all nodes
        self.node_embeddings = {}
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Pre-compute embeddings for all nodes."""
        print("Pre-computing node embeddings...")
        node_texts = []
        node_ids = []
        
        for node_id, node in self.kg.nodes.items():
            node_texts.append(node.to_text())
            node_ids.append(node_id)
        
        embeddings = self.encoder.encode(node_texts)
        
        for node_id, embedding in zip(node_ids, embeddings):
            self.node_embeddings[node_id] = embedding
        
        print(f"Computed embeddings for {len(node_ids)} nodes")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Node, float]]:
        """Retrieve top-k most relevant nodes for query."""
        query_embedding = self.encoder.encode([query])[0]
        
        # Compute similarities
        similarities = []
        for node_id, node_embedding in self.node_embeddings.items():
            similarity = np.dot(query_embedding, node_embedding)
            similarities.append((node_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k nodes with scores
        results = []
        for node_id, score in similarities[:top_k]:
            if score > 0:  # Only return relevant nodes
                results.append((self.kg.get_node(node_id), score))
        
        return results
    
    def generate_response(self, query: str, retrieved_nodes: List[Tuple[Node, float]]) -> str:
        """Generate response using retrieved nodes and graph context."""
        if not retrieved_nodes:
            return "I couldn't find relevant information in the knowledge graph."
        
        main_node, main_score = retrieved_nodes[0]
        
        # Get connected nodes for additional context
        connected = self.kg.get_connected_nodes(main_node.id, depth=2)
        
        # Build response based on node type
        response_parts = ["Based on the university knowledge graph:\n"]
        
        # Main information
        props = main_node.properties
        if main_node.type == "university":
            response_parts.append(
                f"{props['name']} was founded in {props.get('founded', 'unknown')}."
            )
        
        elif main_node.type == "college":
            response_parts.append(
                f"{props['name']} is led by {props.get('dean', 'unknown')}."
            )
            
            # Add majors offered
            majors = [n for n in connected if n.type == "major"]
            if majors:
                major_names = [n.properties['name'] for n in majors]
                response_parts.append(
                    f"This college offers {len(majors)} majors: {', '.join(major_names)}."
                )
        
        elif main_node.type == "major":
            response_parts.append(
                f"{props['name']} is a major with {props.get('students', 'unknown')} students."
            )
            
            # Find parent college
            college = next((n for n in connected if n.type == "college"), None)
            if college:
                response_parts.append(
                    f"It is offered by {college.properties['name']}."
                )
        
        elif main_node.type == "facility":
            response_parts.append(
                f"{props['name']} is open {props.get('hours', 'unknown')}."
            )
            if 'capacity' in props:
                response_parts.append(f"Capacity: {props['capacity']} students.")
            if 'floors' in props:
                response_parts.append(f"It has {props['floors']} floors.")
        
        # Add related nodes
        if len(retrieved_nodes) > 1:
            related_names = [n.properties.get('name', n.id) for n, _ in retrieved_nodes[1:3]]
            response_parts.append(f"\nRelated: {', '.join(related_names)}.")
        
        return "\n".join(response_parts)
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Main RAG pipeline: retrieve + generate."""
        print(f"\n{'='*60}")
        print(f"Query: {query_text}")
        print(f"{'='*60}")
        
        # Retrieve relevant nodes
        retrieved = self.retrieve(query_text, top_k=top_k)
        
        print(f"\nRetrieved {len(retrieved)} nodes:")
        for node, score in retrieved:
            print(f"  - {node.properties.get('name', node.id)} ({node.type}): {score:.3f}")
        
        # Generate response
        response = self.generate_response(query_text, retrieved)
        
        print(f"\nResponse:\n{response}")
        
        return {
            "query": query_text,
            "retrieved_nodes": [
                {
                    "id": node.id,
                    "type": node.type,
                    "properties": node.properties,
                    "score": float(score)
                }
                for node, score in retrieved
            ],
            "response": response
        }


def create_university_kg() -> KnowledgeGraph:
    """Create sample university knowledge graph."""
    kg = KnowledgeGraph()
    
    # University
    kg.add_node(Node("uni", "university", {
        "name": "Tech State University",
        "founded": 1905,
        "location": "California"
    }))
    
    # Colleges
    kg.add_node(Node("eng", "college", {
        "name": "College of Engineering",
        "dean": "Dr. Sarah Chen"
    }))
    kg.add_node(Node("las", "college", {
        "name": "College of Liberal Arts",
        "dean": "Dr. Michael Torres"
    }))
    kg.add_node(Node("bus", "college", {
        "name": "School of Business",
        "dean": "Dr. Jennifer Park"
    }))
    
    # Engineering majors
    kg.add_node(Node("cs", "major", {
        "name": "Computer Science",
        "students": 450,
        "degree_type": "BS"
    }))
    kg.add_node(Node("ee", "major", {
        "name": "Electrical Engineering",
        "students": 320,
        "degree_type": "BS"
    }))
    kg.add_node(Node("me", "major", {
        "name": "Mechanical Engineering",
        "students": 380,
        "degree_type": "BS"
    }))
    
    # Liberal Arts majors
    kg.add_node(Node("psych", "major", {
        "name": "Psychology",
        "students": 520,
        "degree_type": "BA"
    }))
    kg.add_node(Node("eng_lit", "major", {
        "name": "English Literature",
        "students": 280,
        "degree_type": "BA"
    }))
    
    # Business majors
    kg.add_node(Node("finance", "major", {
        "name": "Finance",
        "students": 410,
        "degree_type": "BBA"
    }))
    kg.add_node(Node("mkt", "major", {
        "name": "Marketing",
        "students": 350,
        "degree_type": "BBA"
    }))
    
    # Facilities
    kg.add_node(Node("lib", "facility", {
        "name": "Main Library",
        "hours": "24/7",
        "floors": 5
    }))
    kg.add_node(Node("gym", "facility", {
        "name": "Student Recreation Center",
        "hours": "6am-11pm"
    }))
    kg.add_node(Node("dorm1", "facility", {
        "name": "North Campus Housing",
        "capacity": 800
    }))
    
    # Edges - College relationships
    kg.add_edge(Edge("eng", "uni", "part_of"))
    kg.add_edge(Edge("las", "uni", "part_of"))
    kg.add_edge(Edge("bus", "uni", "part_of"))
    
    # Major relationships
    kg.add_edge(Edge("cs", "eng", "offered_by"))
    kg.add_edge(Edge("ee", "eng", "offered_by"))
    kg.add_edge(Edge("me", "eng", "offered_by"))
    
    kg.add_edge(Edge("psych", "las", "offered_by"))
    kg.add_edge(Edge("eng_lit", "las", "offered_by"))
    
    kg.add_edge(Edge("finance", "bus", "offered_by"))
    kg.add_edge(Edge("mkt", "bus", "offered_by"))
    
    # Facility relationships
    kg.add_edge(Edge("lib", "uni", "facility_of"))
    kg.add_edge(Edge("gym", "uni", "facility_of"))
    kg.add_edge(Edge("dorm1", "uni", "facility_of"))
    
    return kg


def main():
    """Main demo function."""
    print("Knowledge Graph RAG Proof of Concept")
    print("=" * 60)
    
    # Create knowledge graph
    print("\nCreating university knowledge graph...")
    kg = create_university_kg()
    print(f"Created KG with {len(kg.nodes)} nodes and {len(kg.edges)} edges")
    
    # Initialize encoder (try sentence-transformers, fallback to simple)
    print("\nInitializing encoder...")
    try:
        encoder = SentenceTransformerEncoder("all-MiniLM-L6-v2")
    except (ImportError, Exception) as e:
        print(f"Using simple encoder (install sentence-transformers for better results)")
        encoder = SimpleEncoder()
    
    # Initialize RAG system
    rag = KnowledgeGraphRAG(kg, encoder)
    
    # Example queries
    queries = [
        "Computer Science major",
        "engineering programs",
        "library hours",
        "School of Business dean",
        "psychology students",
        "Largest College"
    ]
    
    print("\n" + "=" * 60)
    print("Running example queries...")
    print("=" * 60)
    
    for query in queries:
        result = rag.query(query)
        print()  # Extra spacing


if __name__ == "__main__":
    main()