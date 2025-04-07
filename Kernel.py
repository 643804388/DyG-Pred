from graphkernels import RandomWalkKernel
from graphkernels import GraphletKernel
kernel = GraphletKernel()

similarity_score = kernel.fit_transform([graph1, graph2])
kernel = RandomWalkKernel()
similarity_score = kernel.fit_transform([graph1, graph2])