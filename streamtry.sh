hadoop \
jar /opt/cloudera/parcels/CDH-5.16.1-1.cdh5.16.1.p0.3/lib/hadoop-mapreduce/hadoop-streaming.jar \
-mapper mapper.py \
-reducer reducer.py \
-numReduceTasks 12 \
-input /groups/sri/itan/split_encounters/hourly.tsv \
-output /groups/sri/itan/split_encounters/hourly_output \
-file /home/i016983/Development/pymap/mapper.py \
-file /home/i016983/Development/pymap/reducer.py
