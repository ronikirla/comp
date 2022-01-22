from array import array
import sys
import xml.etree.ElementTree as ET
from decimal import Decimal, DecimalException
from datetime import timedelta
import random

def duration(duration_string):
    duration_string = duration_string.lower()
    total_seconds = Decimal('0')
    prev_num = []
    mode = "h"
    for character in duration_string:
        if character  == ":":
            if prev_num:
                num = Decimal(''.join(prev_num))
                if mode == 'h':
                    total_seconds += num * 60 * 60
                    mode = "m"
                elif mode == 'm':
                    total_seconds += num * 60
                prev_num = []
        elif character.isnumeric() or character == '.':
            prev_num.append(character)
    
    return timedelta(seconds=float(total_seconds + Decimal(''.join(prev_num))))

if "--help" in sys.argv:
    print("""
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
""")
    exit()

try:
    file = sys.argv[1]
    goal = duration(sys.argv[2])
except IndexError:
    print("Missing file path or goal time")
    exit()
try:
    weight_mul = 0.75
    if "-w" in sys.argv:
        try:
            weight_mul = float(sys.argv[sys.argv.index("-w") + 1])
        except IndexError:
            weight_mul = 0.75
    tree = ET.parse(file)
    root = tree.getroot()
    segments = []
    skips_prev = []
    for segment in root.find("Segments").findall("Segment"):
        weight = 1
        splits = []
        skips = []
        for split in reversed(segment.find("SegmentHistory").findall("Time")):
            time = split.find("RealTime")
            if time != None and split.attrib["id"] not in skips_prev:
                splits.append((duration(time.text), weight))
                if "--linear" in sys.argv:
                    weight -= 1 / len(segment.find("SegmentHistory").findall("Time"))
                else:
                    weight *= weight_mul
            elif time == None:
                skips.append(split.attrib["id"])
        skips_prev = skips
        splits.sort(key=lambda x: x[0])
        weight_sum = sum(map(lambda x: x[1], splits))
        segments.append(list(map(lambda x: (x[0], x[1] / weight_sum), splits)))

    def time_at_percentile(splits, percentile):
        if percentile <= splits[0][1] / 2:
            return splits[0][0]
        elif percentile >= 1 - splits[-1][1] / 2:
            return splits[-1][0]
        else:
            prev = splits[0]
            accuml = prev[1] / 2
            for split in splits[1:]:
                step = (prev[1] + split[1]) / 2
                if percentile >= accuml and percentile <= accuml + step:
                    return prev[0] + (split[0] - prev[0]) * (percentile - accuml) / step
                accuml += step
                prev = split
    
    def finish_at_percentile(percentile):
        return sum(map(lambda x: time_at_percentile(x, percentile), segments), timedelta(seconds=0))
    
    def find_goal_splits(goal):
        def search(min, max):
            percentile = (min + max) / 2
            result = finish_at_percentile(percentile)
            if abs(result.total_seconds() - goal.total_seconds()) < 0.1:
                return percentile
            elif result > goal:
                return search(min, max - (max - min) / 2)
            elif result < goal:
                return search(min + (max - min) / 2, max)
     
        percentile = search(0, 1)
        print("Found percentile:", percentile)
        time = timedelta(seconds=0)
        for splits in segments[:-1]:
            time += time_at_percentile(splits, percentile)
            print(time)
        print(goal)

    def simulate_runs(start_split, start_time, goal, reset_times = [], target_percentage = None):
        count = 0
        success = 0
        convergence = 0
        stored_times = [[None] * 101 for i in range(len(segments))]
        def percentage():
            return success / count * 100
        prev_percentage = -1
        if "--chunk" in sys.argv:
            chunk_size = int(sys.argv[sys.argv.index("--chunk") + 1])
        else:
            chunk_size = 1
        while True:
            count += 1
            sum = start_time
            skip = start_split
            reset = False
            reroll_percentile = chunk_size
            percentile = round(random.random(), 2)
            for idx, splits in enumerate(segments):
                if skip <= 0:
                    if stored_times[idx][int(percentile * 100)] != None:
                        sum += stored_times[idx][int(percentile * 100)]
                    else:
                        time = time_at_percentile(splits, percentile)
                        sum += time
                        stored_times[idx][int(percentile * 100)] = time
                    if idx < len(reset_times) and reset_times[idx] != None and sum > reset_times[idx]:
                        reset = True
                        break
                    reroll_percentile -= 1
                    if reroll_percentile == 0:
                        percentile = round(random.random(), 2)
                        reroll_percentile = chunk_size
                skip -= 1
            if reset == False and sum < goal:
                success += 1
            if target_percentage != None:
                if count < 10000:
                    continue
                elif percentage() <= target_percentage * 0.5 or percentage() >= target_percentage * 1.5:
                    break
                elif percentage() <= prev_percentage * 1.001 and percentage() >= prev_percentage * 0.999:
                    convergence += 1
                    if convergence >= 500:
                        break
                else:
                    prev_percentage = percentage()
                    convergence = 0
            else:
                if count < 10000:
                    continue
                elif percentage() <= prev_percentage * 1.001 and percentage() >= prev_percentage * 0.999:
                    convergence += 1
                    if convergence >= 1000:
                        break
                else:
                    prev_percentage = percentage()
                    convergence = 0
        print(f"{percentage()}%")
        return percentage()

    def find_reset_splits(goal):
        try:
            max_iterations = float(sys.argv[sys.argv.index("--reset") + 1])
        except:
            max_iterations = 1
        times = [None] * (len(segments) - 1)
        iterations = 0
        while True:
            iterations += 1
            times_prev = times.copy()
            base_percentage = simulate_runs(0, timedelta(seconds=0), goal, times_prev)
            if base_percentage < 0.1:
                print("Warning: Too low odds, estimates may be wrong")
            for idx, start_split in enumerate(range(1, len(segments))):
                print(f"Split: {start_split} / {len(segments)}")
                def search(min, max):
                    if round(timedelta.total_seconds(min)) == round(timedelta.total_seconds(max)):
                        return min
                    start_time = (min + max) / 2
                    print(f"Time: {start_time}")
                    run_progress_factor = goal / (goal - start_time)
                    def repeated_odds(p0, n):
                        return (1 - pow(1 - p0 / 100, n)) * 100
                    result = repeated_odds(simulate_runs(start_split, start_time, goal, times_prev, repeated_odds(base_percentage, run_progress_factor)), run_progress_factor)
                    try:
                        if result <= base_percentage * 1.01 and result >= base_percentage * 0.99:
                            return start_time
                        elif result < base_percentage:
                            return search(min, max - (max - min) / 2)
                        elif result > base_percentage:
                            return search(min + (max - min) / 2, max)
                    except RecursionError:
                        print("Warning: Did not find the right odds")
                        return start_time
                times[idx] = search(timedelta(seconds=0), goal)
            print(f"Iterations: {iterations}")
            for time in times:
                print(time)
            print(goal)


    if "--sim" in sys.argv:
        try:
            start_split = float(sys.argv[sys.argv.index("--sim") + 1])
        except:
            start_split = 0
        try:
            start_time = duration(sys.argv[sys.argv.index("--sim") + 2])
        except:
            start_time = timedelta(seconds=0)
        simulate_runs(start_split, start_time, goal)
    elif "--reset" in sys.argv:
        find_reset_splits(goal)
    else:
        find_goal_splits(goal)

except FileNotFoundError as err:
    print("File not found:", err.filename)
except ET.ParseError:
    print("Malformatted splits")
except RecursionError:
    print("Could not find the target percentile, check your goal time")