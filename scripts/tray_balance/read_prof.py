import pstats
import IPython

stats = pstats.Stats("prof.stats")
stats.strip_dirs()
stats.sort_stats("cumulative")
stats.print_stats()
