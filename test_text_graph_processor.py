import pytest
import os
import tempfile
from text_graph_processor import TextGraphProcessor

@pytest.fixture
def text_graph_processor():
    """Fixture to create and initialize a TextGraphProcessor with test data from a temporary file."""
    # 创建临时文件并写入测试文本
    test_text = (
        "The scientist carefully analyzed the data, wrote a detailed report, "
        "and shared the report with the team, but the team requested more data, "
        "so the scientist analyzed it again."
    )

    # 使用临时文件
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        temp_file.write(test_text)
        temp_file_path = temp_file.name

    # 创建处理器并加载文件
    processor = TextGraphProcessor()
    result = processor.build_graph_from_file(temp_file_path)
    if not result:
        pytest.fail("Failed to build graph from test file.")
    
    # 打印图的信息进行调试
    print("Graph nodes:", processor.graph.nodes())
    print("Graph edges:", processor.graph.edges(data=True))

    yield processor  # 这是测试使用的处理器实例

    # 测试完成后清理临时文件
    try:
        os.unlink(temp_file_path)
    except:
        pass

class TestFindBridgeWords:

    def test_no_bridge_words_within_hops(self, text_graph_processor):
        # Test case 2: word1="analyzed", word2="data", max_hops=1
        bridge_words, error_msg = text_graph_processor.find_bridge_words("analyzed", "data", 1)
        assert bridge_words == []
        assert "No bridge words from 'analyzed' to 'data' within 1 hops" in error_msg

    def test_multiple_bridge_words_within_hops(self, text_graph_processor):
        # Test case 3: word1="scientist", word2="report", max_hops=3
        bridge_words, error_msg = text_graph_processor.find_bridge_words("scientist", "data", 3)
        print("Actual bridge words:", bridge_words)  # 打印实际的桥接词
        assert set(bridge_words) == {"analyzed", "the"}
        assert error_msg is None


    def test_bridge_words_across_sentence(self, text_graph_processor):
        # Test case 4: word1="team", word2="data", max_hops=3
        bridge_words, error_msg = text_graph_processor.find_bridge_words("team", "data", 3)
        assert set(bridge_words) == {"requested", "more", "but", "the"}
        assert error_msg is None

    def test_long_path_bridge_words(self, text_graph_processor):
        # Test case 5: word1="the", word2="again", max_hops=6
        bridge_words, error_msg = text_graph_processor.find_bridge_words("the", "again", 6)
        assert set(bridge_words) == {"scientist", "analyzed", "it", "carefully"}
        assert error_msg is None

    def test_word1_not_in_graph(self, text_graph_processor):
        # Test case 6: word1="unknown", word2="team", max_hops=2
        bridge_words, error_msg = text_graph_processor.find_bridge_words("unknown", "team", 2)
        assert bridge_words is None
        assert "No 'unknown' or 'team' in the graph" in error_msg

    def test_word2_not_in_graph(self, text_graph_processor):
        # Test case 7: word1="report", word2="unknown", max_hops=2
        bridge_words, error_msg = text_graph_processor.find_bridge_words("report", "unknown", 2)
        assert bridge_words is None
        assert "No 'report' or 'unknown' in the graph" in error_msg

    def test_same_word_no_bridge(self, text_graph_processor):
        # Test case 8: word1="data", word2="data", max_hops=2
        bridge_words, error_msg = text_graph_processor.find_bridge_words("data", "data", 2)
        assert bridge_words == []
        assert "No bridge words from 'data' to 'data' within 2 hops" in error_msg

    def test_no_bridge_words_within_limited_hops(self, text_graph_processor):
        # Test case 9: word1="team", word2="data", max_hops=1
        bridge_words, error_msg = text_graph_processor.find_bridge_words("team", "data", 1)
        assert bridge_words == []
        assert "No bridge words from 'team' to 'data' within 1 hops" in error_msg