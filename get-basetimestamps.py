# /usr/bin/python3
import os
import argparse

def get_basetimestamps_method1(kml_dir):
    timestamps = []
    for filename in os.listdir(kml_dir):
        if filename.endswith(".kml"):
            # ex: convert "log_4_2024-7-6-16-49-04.kml" to "2024-07-06T16:49:04+00:00"
            time = filename.split('.')[0].split('_')[2]
            year = time.split('-')[0]
            month = int(time.split('-')[1])
            day = int(time.split('-')[2])
            hour = int(time.split('-')[3])
            minute = int(time.split('-')[4])
            second = int(time.split('-')[5])
            timestamp = year + '-' + f'{month:02}' + '-' + f'{day:02}' + 'T' + f'{hour:02}' + ':' + f'{minute:02}' + ':' + f'{second:02}' + '+00:00'
            timestamps.append(timestamp)

    # sort timestamps with increasing order based on year, month, day, hour, minute, second accordingly
    timestamps.sort()
                
    return timestamps

def get_basetimestamps_method2(kml_dir):
    timestamps = []
    for filename in os.listdir(kml_dir):
        if filename.endswith(".kml"):
            # ex: convert "飛行軌跡_20230716070634_R8039938565.kml" to "2023-07-16T07:06:34+00:00"
            time = filename.split('.')[0].split('_')[1]
            year = time[0:4]
            month = int(time[4:6])
            day = int(time[6:8])
            hour = int(time[8:10])
            minute = int(time[10:12])
            second = int(time[12:14])
            timestamp = year + '-' + f'{month:02}' + '-' + f'{day:02}' + 'T' + f'{hour:02}' + ':' + f'{minute:02}' + ':' + f'{second:02}' + '+00:00'
            timestamps.append(timestamp)

    # sort timestamps with increasing order based on year, month, day, hour, minute, second accordingly
    timestamps.sort()
                
    return timestamps

def main(args):
    if args.method == 1:
        timestamps = get_basetimestamps_method1(args.kml_dir)
    elif args.method == 2:
        timestamps = get_basetimestamps_method2(args.kml_dir)
    print(timestamps)

    # write to txt file
    with open(args.csv, 'w') as f: #'./data/ardupilot-logs-timestamps.txt'
        for timestamp in timestamps:
            f.write("%s\n" % timestamp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get timestamps from kml files')
    parser.add_argument('--kml_dir', type=str, default='./data/ardupilot-logs/kmls', help='Directory of kml files')
    parser.add_argument('--method', type=int, default=1, help='Method to get timestamps from kml files')
    parser.add_argument('--csv', type=str, default='./data/ardupilot-logs-timestamps.txt', help='Output file')
    args = parser.parse_args()
    main(args)

# Run the script
# python3 get-basetimestamps.py --kml_dir ./data/ardupilot-logs/kmls --method 1 --csv ./data/ardupilot-logs-timestamps.txt
# python3 get-basetimestamps.py --kml_dir ./data/raw_kml --method 2 --csv ./data/uav-dataset-timestamps.txt


