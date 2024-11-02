import csv
import requests
import sys
import re
from datetime import datetime
from os import mkdir
from jproperties import Properties

# Filter metrics from the config file
def get_metrics_name():
    metrics_file = 'metrics.txt'
    if len(sys.argv) > 4:
        metrics_file = sys.argv[4]

    with open('../Utilities/config/' + metrics_file) as input_metrics:
        lines = input_metrics.read().splitlines()
    new_names = list(set(lines))
    return new_names

# Load settings from a properties file
def load_settings():
    configs = Properties()
    with open('../Utilities/config/settings.properties', 'rb') as config_file:
        configs.load(config_file)
    return configs

# Validate a URL
def check_url(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url)

# Generate a filename based on metric content and result metadata
def generate_filename(metric_name, result, metric_count):
    # Handle filenames for different types of metrics
    if "avg" in metric_name:
        file_name = f"{metric_count}_average.csv"
    elif "replicas" in metric_name:
        file_name = f"{metric_count}_replicas.csv"
    else:
        # Check if there is a specific pod name in the result's 'metric' field
        pod_identifier = result['metric'].get('pod')
        if pod_identifier:
            pod_identifier = pod_identifier.replace("microsvc-", "")
            file_name = f"{metric_count}_microsvc-{pod_identifier}.csv"
        else:
            file_name = f"{metric_count}_metric.csv"  # Fallback filename
    
    # Remove invalid characters for filenames
    file_name = re.sub(r'[<>:"/\\|?*]', '_', file_name)
    return file_name

def main():
    if len(sys.argv) < 4:
        print('Invalid number of arguments, a minimum of 3 arguments: <prometheus_url> <start_date> <end_date>')
        sys.exit(1)

    # Validate Prometheus URL
    prometheus_url = sys.argv[1]
    if check_url(prometheus_url) is None:
        print('Error passing argument <prometheus_url>: Invalid URL format')
        sys.exit(1)

    metric_names = get_metrics_name()
    configs = load_settings()
    write_header = True
    now = datetime.now

    # Create performance directory with timestamp
    ts_title = now().strftime('%Y-%m-%d-%H-%M-%S')
    new_folder = '../Utilities/csv/metrics_' + ts_title
    mkdir(new_folder)

    metric_count = 1
    for metric_name in metric_names:
        print('Exporting metric name:' + metric_name)
        response = requests.get(
            '{0}/api/v1/query_range'.format(prometheus_url),
            params={
                'query': metric_name, 'start': sys.argv[2], 'end': sys.argv[3], 'step': '60s'
            },
            verify=False
        )
        results = response.json()['data']['result']

        for result in results:
            # Generate filename with the specific pod name from `result`
            csv_file_name = new_folder + "/" + generate_filename(metric_name, result, metric_count)
            metric_count += 1

            # Write to CSV file
            with open(csv_file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                if write_header:
                    writer.writerow(['datetime', 'value'])
                    write_header = False
                str_new = str(result['values']).replace("[", "").replace("]", "")
                value_array = str_new.split(",")
                index = 0
                while index < len(value_array) - 1:
                    subl = []
                    ts_value = value_array[index]
                    t = datetime.utcfromtimestamp(float(ts_value))
                    metric_value = value_array[index + 1].replace("'", "").replace("u", "").strip()
                    index += 2
                    subl.append(t.strftime("%d/%m/%Y %H:%M:%S"))
                    subl.append(metric_value)
                    writer.writerow(subl)
                write_header = True

if __name__ == "__main__":
    main()