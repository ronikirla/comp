# comp

This script can be used to create balanced LiveSplit comparisons with customizable parameters, such as goal time,
recency weighing and so on.
Outputs split times in text that can be copy-pasted directly into a custom LiveSplit comparison.

Usage: python <path to the script> <path to your splits file> <target time> [optional args]
Target time should be given as hh:mm:ss.
Examples: python comp.py C:\splits.lss 0:12:34 --linear
          python comp.py C:\splits.lss 0:12:34 -w 0.9 --reset

Optional args: 
    -w <value>
        Recency weighing with geometric decay. Recommended if you have a small amount of valid data, for example after
        finding a new faster route or improving your play significantly.
        If no weighing method is specified, this is used by default with a value of 0.75 (LiveSplit default).
    --linear
        Recency weighing with a linear decay, most recent split getting weight 1 and oldest getting weight 0.
        Recommended in the long run, when the majority of the splits data is collected with your current route.
    --sim [start split] [start time]
        Simulates runs using the existing split data. Outputs the odds of beating the goal time.
        Starts the simulation on the start split number at the start time if specified.
    --reset <number of iterations>
        Creates a custom comparison where it is equally likely to hit the goal time from the current split
        as it is from a reset.
        Accounts for the IRL time investment, so runs progressed further require lower odds of succeeding.
        You can use this to gauge when it makes mathematical sense to reset -- if you're ahead this there
        should be no reason to reset. Uses simulations and as such takes a while.
        Runs a specified amount of iterations (default 1) where the times of the previous iteration are used
        as reset point on the current one.
        Use ctrl+c to exit.
    --chunk <size>
        Bundles up segments into chunks that are given the same percentile when simulating runs.
        This should speed up the process at the cost of accuracy. Not recommended.
