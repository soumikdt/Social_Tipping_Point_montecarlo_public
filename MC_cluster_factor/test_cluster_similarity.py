"""
Quick test script to verify cluster similarity function works correctly
"""
import sys
sys.path.insert(0, '../src')
import networkx as nx
import numpy as np
from AgentManipulator import AllAgents

def test_cluster_similarity():
    """Test the cluster similarity function"""
    print("="*60)
    print("Testing Cluster Similarity Function")
    print("="*60)
    
    # Create a simple test graph
    G = nx.Graph()
    G.add_node(1, kmeans_cluster=0)
    G.add_node(2, kmeans_cluster=0)  # Same cluster as 1
    G.add_node(3, kmeans_cluster=1)  # Different cluster
    G.add_node(4)  # No cluster attribute
    
    # Test 1: Same cluster
    result1 = AllAgents.quantile_similarity(
        G, 1, 2, 
        same_cluster_factor=1.0, 
        diff_cluster_factor=0.5
    )
    print(f"\nTest 1 - Same cluster (0, 0):")
    print(f"  Expected: 1.0")
    print(f"  Got: {result1}")
    assert result1 == 1.0, "FAILED: Same cluster should return 1.0"
    print("  ✓ PASSED")
    
    # Test 2: Different cluster
    result2 = AllAgents.quantile_similarity(
        G, 1, 3, 
        same_cluster_factor=1.0, 
        diff_cluster_factor=0.5
    )
    print(f"\nTest 2 - Different cluster (0, 1):")
    print(f"  Expected: 0.5")
    print(f"  Got: {result2}")
    assert result2 == 0.5, "FAILED: Different cluster should return 0.5"
    print("  ✓ PASSED")
    
    # Test 3: Missing cluster attribute
    result3 = AllAgents.quantile_similarity(
        G, 1, 4, 
        same_cluster_factor=1.0, 
        diff_cluster_factor=0.5
    )
    print(f"\nTest 3 - Missing cluster attribute:")
    print(f"  Expected: 1.0 (no effect)")
    print(f"  Got: {result3}")
    assert result3 == 1.0, "FAILED: Missing cluster should return 1.0"
    print("  ✓ PASSED")
    
    # Test 4: Different factors
    result4 = AllAgents.quantile_similarity(
        G, 1, 3, 
        same_cluster_factor=1.0, 
        diff_cluster_factor=0.3
    )
    print(f"\nTest 4 - Different cluster with factor 0.3:")
    print(f"  Expected: 0.3")
    print(f"  Got: {result4}")
    assert result4 == 0.3, "FAILED: Should return 0.3"
    print("  ✓ PASSED")
    
    print("\n" + "="*60)
    print("All tests PASSED! ✓")
    print("="*60)

if __name__ == "__main__":
    test_cluster_similarity()

