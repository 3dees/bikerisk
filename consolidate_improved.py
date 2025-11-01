"""
Improved consolidation logic using Core + Deltas approach.
Replaces the overly aggressive consolidate_ai.py
"""

import anthropic
from typing import Dict, List, Set, Optional, Tuple
import pandas as pd
from rapidfuzz import fuzz
import json
import os
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

@dataclass
class Requirement:
    """Individual requirement from a standard"""
    text: str
    standard: str
    clause: str
    scope: str = ""
    formatting_required: str = ""
    required_in_print: str = ""
    comments: str = ""
    
    @property
    def has_temperature(self) -> bool:
        return '°C' in self.text or '°F' in self.text or 'temperature' in self.text.lower()
    
    @property
    def has_measurements(self) -> bool:
        return bool(re.search(r'\d+\.?\d*\s*(mm|cm|m|inch|")', self.text))
    
    @property
    def has_warning(self) -> bool:
        return any(word in self.text.upper() for word in ['WARNING', 'CAUTION', 'DANGER'])

@dataclass
class CoreDeltaGroup:
    """A consolidated group with core requirement and deltas"""
    group_id: int
    core_requirement: str
    applies_to: List[Dict]  # List of {standard, clause} dicts
    deltas: Dict[str, List[str]]  # Standard -> unique requirements
    similarity_score: float
    topic: str
    requirement_count: int

class ImprovedConsolidator:
    """
    Improved consolidation for BikeRisk that creates manageable groups
    with preserved details using Core + Deltas approach
    """
    
    # Configuration - tuned for bike standards
    MAX_GROUP_SIZE = 12  # Optimal for bike requirements
    MIN_SIMILARITY = 0.75
    MIN_CORE_RATIO = 0.5  # Core must be 50% of each requirement
    
    # Topic definitions for bike/EPAC standards
    TOPIC_KEYWORDS = {
        'assembly': ['assembly', 'assemble', 'installation', 'mounting', 'unassembled'],
        'battery_charging': ['battery', 'charging', 'charger', 'temperature range', 'charge'],
        'maintenance': ['maintenance', 'lubrication', 'adjustment', 'inspection', 'repair'],
        'safety_warnings': ['warning', 'danger', 'caution', 'risk', 'hazard'],
        'manual_provision': ['instruction manual', 'documentation', 'instructions', 'provided with'],
        'language': ['language', 'translation', 'official language'],
        'digital_format': ['digital', 'electronic', 'online', 'downloadable', 'paper format'],
        'composite': ['composite', 'carbon', 'wear', 'damage', 'rim'],
        'tampering': ['tampering', 'modification', 'alter', 'modify'],
        'brakes': ['brake', 'braking', 'friction', 'lever'],
        'electrical': ['electrical', 'wiring', 'cable', 'connection'],
        'grounding': ['ground', 'grounding', 'earth'],
        'symbols': ['symbol', 'marking', 'label', 'pictogram'],
        'tools': ['tool', 'equipment', 'necessary'],
    }
    
    def __init__(self):
        self.groups = []
        self.ungrouped = []
        
    def consolidate_from_dataframe(self, df: pd.DataFrame) -> Dict:
        """
        Main entry point - processes requirements from DataFrame
        Compatible with existing BikeRisk structure
        """
        # Convert DataFrame to Requirement objects
        requirements = self._df_to_requirements(df)
        
        # Process with Core + Deltas logic
        groups = self._process_requirements(requirements)
        
        # Format for BikeRisk output
        return self._format_output(groups, df)
    
    def _df_to_requirements(self, df: pd.DataFrame) -> List[Requirement]:
        """Convert BikeRisk DataFrame to Requirement objects"""
        requirements = []
        
        # ADD THIS DEBUG OUTPUT FIRST:
        print(f"\n[DEBUG] DataFrame has {len(df)} rows")
        print(f"[DEBUG] Column names: {list(df.columns)}")
        print(f"[DEBUG] First row sample:")
        if len(df) > 0:
            first_row = df.iloc[0]
            for col in df.columns:
                print(f"  - {col}: {first_row[col]}")
        
        for idx, row in df.iterrows():
            # Handle different column name variations
            text = (row.get('Requirement (Clause)') or 
                   row.get('Description') or 
                   row.get('requirement_text', ''))
            
            standard = (row.get('Standard/ Regulation') or 
                       row.get('Standard/Reg') or 
                       row.get('standard', ''))
            
            clause = (row.get('Clause ID') or 
                     row.get('Clause/Requirement') or 
                     row.get('Clause', ''))
            
            # ADD THIS DEBUG FOR FIRST FEW ROWS:
            if idx < 3:
                print(f"\n[DEBUG] Row {idx}:")
                print(f"  text found: {text is not None and text != ''}")
                print(f"  text value: {str(text)[:100] if text else 'NONE'}")
                print(f"  standard: {standard}")
                print(f"  clause: {clause}")
            
            if pd.notna(text) and str(text).strip():
                req = Requirement(
                    text=str(text).strip(),
                    standard=str(standard).strip(),
                    clause=str(clause).strip()
                )
                requirements.append(req)
        
        # ADD THIS AT THE END:
        print(f"\n[DEBUG] Total requirements converted: {len(requirements)}")
        
        return requirements
    
    def _process_requirements(self, requirements: List[Requirement]) -> List[CoreDeltaGroup]:
        """Core processing logic"""
        
        # Step 1: Topic clustering
        topic_clusters = self._cluster_by_topic(requirements)
        
        # Step 2: Process each topic cluster
        all_groups = []
        group_id = 0
        
        for topic, reqs in topic_clusters.items():
            if len(reqs) < 2:
                self.ungrouped.extend(reqs)
                continue
            
            # Split large clusters
            if len(reqs) > self.MAX_GROUP_SIZE:
                subclusters = self._split_large_cluster(reqs)
            else:
                subclusters = [reqs]
            
            # Try to consolidate each subcluster
            for subcluster in subclusters:
                group = self._create_consolidation_group(subcluster, group_id, topic)
                if group:
                    all_groups.append(group)
                    group_id += 1
                else:
                    self.ungrouped.extend(subcluster)

        print(f"\n[DEBUG] Processing complete:")
        print(f"  - Total groups created: {len(all_groups)}")
        print(f"  - Ungrouped requirements: {len(self.ungrouped)}")

        return all_groups
    
    def _cluster_by_topic(self, requirements: List[Requirement]) -> Dict[str, List[Requirement]]:
        """Group requirements by primary topic"""
        clusters = defaultdict(list)
        
        for req in requirements:
            text_lower = req.text.lower()
            assigned = False
            
            # Check each topic
            for topic, keywords in self.TOPIC_KEYWORDS.items():
                if any(kw in text_lower for kw in keywords):
                    clusters[topic].append(req)
                    assigned = True
                    break
            
            if not assigned:
                clusters['general'].append(req)

        print(f"\n[DEBUG] Topic clustering results:")
        for topic, reqs in clusters.items():
            print(f"  - {topic}: {len(reqs)} requirements")

        return dict(clusters)
    
    def _split_large_cluster(self, requirements: List[Requirement]) -> List[List[Requirement]]:
        """Split large clusters into smaller, highly similar groups"""
        subclusters = []
        processed = set()
        
        for i, req1 in enumerate(requirements):
            if i in processed:
                continue
                
            subcluster = [req1]
            processed.add(i)
            
            # Find similar requirements
            for j, req2 in enumerate(requirements[i+1:], start=i+1):
                if j in processed:
                    continue
                    
                if len(subcluster) >= self.MAX_GROUP_SIZE:
                    break
                    
                # Calculate similarity
                similarity = self._calculate_similarity(req1, req2)
                if similarity >= self.MIN_SIMILARITY:
                    subcluster.append(req2)
                    processed.add(j)
            
            if len(subcluster) >= 2:
                subclusters.append(subcluster)
        
        return subclusters
    
    def _calculate_similarity(self, req1: Requirement, req2: Requirement) -> float:
        """Calculate similarity between two requirements"""
        
        # Text similarity using fuzzy matching
        text_sim = fuzz.token_sort_ratio(req1.text, req2.text) / 100.0
        
        # Standard family bonus
        standard_bonus = 0.1 if self._same_standard_family(req1.standard, req2.standard) else 0
        
        # Technical specs similarity
        tech_bonus = 0.1 if (req1.has_temperature == req2.has_temperature and 
                            req1.has_measurements == req2.has_measurements) else 0
        
        # Warning level similarity  
        warning_bonus = 0.1 if req1.has_warning == req2.has_warning else 0
        
        return min(text_sim + standard_bonus + tech_bonus + warning_bonus, 1.0)
    
    def _same_standard_family(self, std1: str, std2: str) -> bool:
        """Check if standards are from same family"""
        families = {
            'bicycle': ['16 CFR', 'EN 15194', 'ISO 4210'],
            'electrical': ['UL 2849', '47 CFR', 'FCC'],
            'machinery': ['MD', 'MR'],
        }
        
        for family, standards in families.items():
            if any(s in std1 for s in standards) and any(s in std2 for s in standards):
                return True
        return False
    
    def _create_consolidation_group(self, requirements: List[Requirement],
                                   group_id: int, topic: str) -> Optional[CoreDeltaGroup]:
        """Create a Core + Deltas consolidation group"""

        print(f"\n[DEBUG] Group {group_id} ({topic}): Trying to consolidate {len(requirements)} requirements")

        if len(requirements) < 2:
            return None
        
        # Extract core requirement
        core = self._extract_core(requirements)
        if not core:
            return None
        
        # Check core ratio
        for req in requirements:
            core_words = set(core.lower().split())
            req_words = set(req.text.lower().split())
            if len(core_words & req_words) / len(req_words) < self.MIN_CORE_RATIO:
                return None
        
        # Extract deltas
        deltas = self._extract_deltas(requirements, core)
        
        # Calculate average similarity
        total_sim = 0
        count = 0
        for i in range(len(requirements)):
            for j in range(i+1, len(requirements)):
                total_sim += self._calculate_similarity(requirements[i], requirements[j])
                count += 1
        
        avg_similarity = total_sim / count if count > 0 else 0
        
        if avg_similarity < self.MIN_SIMILARITY:
            return None
        
        # Create group
        return CoreDeltaGroup(
            group_id=group_id,
            core_requirement=core,
            applies_to=[{'standard': req.standard, 'clause': req.clause} 
                       for req in requirements],
            deltas=deltas,
            similarity_score=avg_similarity,
            topic=topic,
            requirement_count=len(requirements)
        )
    
    def _extract_core(self, requirements: List[Requirement]) -> str:
        """Extract common core elements"""
        
        # Find common meaningful words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'be', 'is', 'are', 'was', 'were'}
        
        word_sets = []
        for req in requirements:
            words = set(req.text.lower().split()) - stop_words
            word_sets.append(words)
        
        # Find intersection
        common_words = word_sets[0]
        for word_set in word_sets[1:]:
            common_words &= word_set
        
        if len(common_words) < 3:
            return ""
        
        # Build core statement
        # Look for common patterns in original requirements
        if all('shall' in req.text.lower() for req in requirements):
            core = "Instructions shall "
        else:
            core = "The manufacturer shall provide "
        
        # Add common concepts
        if 'assembly' in common_words:
            core += "assembly instructions "
        if 'maintenance' in common_words:
            core += "maintenance procedures "
        if 'battery' in common_words:
            core += "battery specifications "
        if 'warning' in common_words:
            core += "safety warnings "
        
        # Add common requirements
        if all('include' in req.text.lower() for req in requirements):
            core += "including necessary details for safe use"
        
        return core.strip() + "."
    
    def _extract_deltas(self, requirements: List[Requirement], core: str) -> Dict[str, List[str]]:
        """Extract unique elements for each requirement"""
        
        deltas = {}
        core_lower = core.lower()
        
        for req in requirements:
            unique_elements = []
            
            # Temperature specifics
            if req.has_temperature:
                temp_match = re.findall(r'[-\d]+\s*°[CF]', req.text)
                if temp_match:
                    unique_elements.append(f"Temperature: {', '.join(temp_match)}")
            
            # Measurements
            if req.has_measurements:
                measure_match = re.findall(r'\d+\.?\d*\s*(mm|cm|m|inch|")', req.text)
                if measure_match:
                    unique_elements.append(f"Measurements: {', '.join(measure_match)}")
            
            # Format requirements
            if 'paper' in req.text.lower() and 'paper' not in core_lower:
                unique_elements.append("Paper format required")
            if 'digital' in req.text.lower() and 'digital' not in core_lower:
                unique_elements.append("Digital format allowed")
            
            # Warning specifics
            if req.has_warning:
                for word in ['WARNING', 'CAUTION', 'DANGER']:
                    if word in req.text.upper():
                        unique_elements.append(f"Uses {word}")
                        break
            
            # Print requirement
            if req.required_in_print == 'y':
                unique_elements.append("Required in print")
            
            # Store deltas
            if unique_elements:
                key = f"{req.standard} {req.clause}".strip()
                deltas[key] = unique_elements
        
        return deltas
    
    def _format_output(self, groups: List[CoreDeltaGroup], original_df: pd.DataFrame) -> Dict:
        """Format output for BikeRisk compatibility"""
        
        output_rows = []
        
        for group in groups:
            # Create main consolidation row
            row = {
                'Group ID': group.group_id,
                'Similarity Score': group.similarity_score,
                'Topic Keywords': group.topic,
                'Reasoning': f"These {group.requirement_count} requirements share core intent about {group.topic}",
                'Suggested Consolidation': group.core_requirement,
                'Original Requirement Indices': [i for i, _ in enumerate(group.applies_to)]
            }
            
            # Add delta information
            if group.deltas:
                delta_summary = []
                for std, deltas in group.deltas.items():
                    delta_summary.append(f"{std}: {'; '.join(deltas)}")
                row['Critical Differences'] = delta_summary
            else:
                row['Critical Differences'] = []
            
            output_rows.append(row)
        
        # Create DataFrame
        result_df = pd.DataFrame(output_rows)
        
        return {
            'consolidation_groups': result_df,
            'statistics': {
                'original_requirements': len(original_df),
                'consolidated_groups': len(groups),
                'avg_group_size': sum(g.requirement_count for g in groups) / len(groups) if groups else 0,
                'ungrouped_count': len(self.ungrouped)
            }
        }
