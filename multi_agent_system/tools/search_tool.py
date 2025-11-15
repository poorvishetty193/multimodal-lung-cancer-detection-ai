# multi-agent-system/tools/search_tool.py
import logging
logger = logging.getLogger(__name__)

class SearchTool:
    """
    Search tool stub — you can wire this to a web.run or Google Search API for grounding.
    """
    def search(self, query: str, num_results: int = 3):
        logger.info("SearchTool.search called with query: %s", query)
        # return lightweight dummy
        return [{"title": "No search configured", "snippet": "Enable search tool to populate results"}]
