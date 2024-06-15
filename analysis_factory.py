from analysis.correlation_analysis import CorrelationAnalysis
from analysis.cluster_analysis import ClusteringAnalysis
from analysis.stat_analysis import StatAnalysis

# Import other analysis classes as needed


class AnalysisFactory:
    """
    Factory class to create different types of analysis objects.
    """

    def create_analysis(self, analysis_type):
        """
        Create an analysis object based on the type.

        Parameters:
        analysis_type (str): The type of analysis to create.

        Returns:
        object: The analysis object.
        """
        if analysis_type == "stat":
            return StatAnalysis()
        elif analysis_type == "cluster":
            return ClusteringAnalysis()
        elif analysis_type == "correlation":
            return CorrelationAnalysis()
        else:
            raise ValueError("Unknown analysis type")


# Example usage
factory = AnalysisFactory()
analysis = factory.create_analysis("correlation")
