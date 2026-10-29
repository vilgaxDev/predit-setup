"""
kg_tooling.py
============
Knowledge Graph tooling with rdflib for proper RDF triple management.
Supports RDF serialization, SPARQL queries, and triple store integration.
"""

from rdflib import Graph, URIRef, Literal, Namespace, RDF
from rdflib.namespace import XSD
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json

class KnowledgeGraphManager:
    """Manages knowledge graph with rdflib for proper RDF operations."""
    
    def __init__(self, namespace_uri: str = "http://ontology.example.org/mch#"):
        """
        Initialize KG manager.
        
        Args:
            namespace_uri: Base URI for the ontology
        """
        self.graph = Graph()
        self.namespace = Namespace(namespace_uri)
        self.graph.bind("mch", self.namespace)
        
        # RDF namespaces
        self.rdf = RDF
        self.xsd = XSD
    
    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        obj_type: str = "literal"
    ) -> None:
        """
        Add a triple to the knowledge graph.
        
        Args:
            subject: Subject URI or identifier
            predicate: Predicate URI or identifier
            obj: Object value
            obj_type: Type of object (literal, uri)
        """
        subj_uri = URIRef(f"{self.namespace}{subject}")
        pred_uri = URIRef(f"{self.namespace}{predicate}")
        
        if obj_type == "uri":
            obj_uri = URIRef(f"{self.namespace}{obj}")
            self.graph.add((subj_uri, pred_uri, obj_uri))
        else:
            obj_literal = Literal(obj)
            self.graph.add((subj_uri, pred_uri, obj_literal))
    
    def add_triples_from_file(
        self,
        file_path: str,
        delimiter: str = "\t"
    ) -> int:
        """
        Load triples from tab-separated file.
        
        Args:
            file_path: Path to triples file
            delimiter: Field delimiter
            
        Returns:
            Number of triples loaded
        """
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(delimiter)
                if len(parts) == 3:
                    subject, predicate, obj = parts
                    self.add_triple(subject, predicate, obj)
                    count += 1
        
        print(f"Loaded {count} triples from {file_path}")
        return count
    
    def export_rdf(self, format: str = "turtle", output_path: str = None) -> str:
        """
        Export graph to RDF format.
        
        Args:
            format: RDF format (turtle, xml, ntriples, json-ld)
            output_path: Output file path
            
        Returns:
            RDF serialization
        """
        rdf_data = self.graph.serialize(format=format)
        
        if output_path:
            if isinstance(rdf_data, bytes):
                with open(output_path, 'wb') as f:
                    f.write(rdf_data)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(rdf_data)
            print(f"RDF exported to {output_path} (format: {format})")
        
        return rdf_data.decode('utf-8') if isinstance(rdf_data, bytes) else rdf_data
    
    def query_sparql(self, query: str) -> List[Tuple]:
        """
        Execute SPARQL query on the knowledge graph.
        
        Args:
            query: SPARQL query string
            
        Returns:
            Query results
        """
        results = self.graph.query(query)
        return list(results)
    
    def get_statistics(self) -> Dict:
        """
        Get knowledge graph statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_triples": len(self.graph),
            "unique_subjects": len(set(self.graph.subjects())),
            "unique_predicates": len(set(self.graph.predicates())),
            "unique_objects": len(set(self.graph.objects()))
        }
    
    def visualize_subgraph(
        self,
        subject: str,
        depth: int = 2
    ) -> Graph:
        """
        Extract subgraph around a subject.
        
        Args:
            subject: Subject URI
            depth: Depth of traversal
            
        Returns:
            Subgraph
        """
        subj_uri = URIRef(f"{self.namespace}{subject}")
        subgraph = Graph()
        
        # Add the subject and its immediate neighbors
        for s, p, o in self.graph.triples((subj_uri, None, None)):
            subgraph.add((s, p, o))
        
        # Add depth-2 neighbors
        for s, p, o in self.graph.triples((None, None, subj_uri)):
            subgraph.add((s, p, o))
        
        return subgraph
    
    def import_from_anonymized_data(
        self,
        df,
        subject_prefix: str = "Patient"
    ) -> int:
        """
        Import anonymized DataFrame into knowledge graph.
        
        Args:
            df: Anonymized DataFrame
            subject_prefix: Prefix for subject URIs
            
        Returns:
            Number of triples imported
        """
        count = 0
        
        for idx, row in df.iterrows():
            subject = f"{subject_prefix}_{idx}"
            
            for col in df.columns:
                value = str(row[col])
                if value and value != "nan":
                    predicate = col.replace(" ", "_").replace("(", "").replace(")", "")
                    self.add_triple(subject, predicate, value)
                    count += 1
        
        print(f"Imported {count} triples from DataFrame")
        return count
    
    def export_to_excel(
        self,
        output_path: str
    ) -> None:
        """
        Export knowledge graph to Excel (structured format).
        
        Args:
            output_path: Output Excel file path
        """
        import pandas as pd
        
        triples_data = []
        for s, p, o in self.graph:
            triples_data.append({
                "Subject": str(s).split("#")[-1],
                "Predicate": str(p).split("#")[-1],
                "Object": str(o)
            })
        
        df = pd.DataFrame(triples_data)
        df.to_excel(output_path, index=False, engine="openpyxl")
        print(f"Exported to Excel: {output_path}")

# Convenience function
def create_kg_from_triples(
    triples_file: str,
    namespace: str = "http://ontology.example.org/mch#"
) -> KnowledgeGraphManager:
    """
    Create knowledge graph from triples file.
    
    Args:
        triples_file: Path to triples file
        namespace: Namespace URI
        
    Returns:
        KnowledgeGraphManager instance
    """
    kg = KnowledgeGraphManager(namespace)
    kg.add_triples_from_file(triples_file)
    return kg
