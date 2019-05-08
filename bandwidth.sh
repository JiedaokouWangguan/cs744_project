# usage run.sh {rank} {bandwidth(xMbps)} {downpoursgd2|easgd}

function terminate_cluster() {
    echo "Terminating the servers"
#    CMD="ps aux | grep -v 'grep' | grep 'python code_template' | awk -F' ' '{print $2}' | xargs kill -9"
    CMD="ps aux | grep -v 'grep' | grep -v 'bash' | grep -v 'ssh' | grep 'python code_template' | awk -F' ' '{print \$2}' | xargs kill -9"
    for i in `seq 0 2`; do
        ssh node$i "$CMD"
    done
}

terminate_cluster
start_time=$(date +%s)


if [ $1 -eq 0 ] 
then 
    arg_ps="--flag"
else 
    arg_ps="--no-flag "                                                                
fi

sudo tc qdisc add dev eno1 handle 1: root htb default 11
sudo tc class add dev eno1 parent 1: classid 1:1 htb rate 1000Mbps
sudo tc class add dev eno1 parent 1:1 classid 1:11 htb rate $2
sudo tc qdisc add dev eno1 parent 1:11 handle 10: netem delay 50ms

dstat -n > network_$3_$2.csv &
python ./$3/main.py --world-size 3 --rank $1 --flag $arg_ps 'tcp://node0:8088' --quantize-nbits 8;

sudo tc qdisc del dev eno1 handle 1: root htb default 11
sudo tc class del dev eno1 parent 1: classid 1:1 htb rate 1000Mbps
sudo tc class del dev eno1 parent 1:1 classid 1:11 htb rate 4Mbit
sudo tc qdisc del dev eno1 parent 1:11 handle 10: netem delay 50ms



end_time=$(date +%s)
diff=$(( $end_time - $start_time ))
echo "It took $diff seconds"
terminate_cluster


