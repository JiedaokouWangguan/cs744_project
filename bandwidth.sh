# usage run.sh {rank} {bandwidth(xMbit)} {downpoursgd2|easgd} {quantize-nbits}

function terminate_cluster() {
    echo "Terminating the servers"
#    CMD="ps aux | grep -v 'grep' | grep 'python code_template' | awk -F' ' '{print $2}' | xargs kill -9"
    ps aux | grep -v 'grep' | grep -v 'bash'| grep -v 'ssh' | grep 'python' | awk -F' ' '{print $2}' | xargs kill -9
}

terminate_cluster
start_time=$(date +%s)


if [ $1 -eq 0 ] 
then 
    arg_ps="--flag"
else 
    arg_ps="--no-flag "                                                                
fi

if [[ $2 != "0" ]]
then
    sudo tc qdisc add dev eno1 handle 1: root htb default 11
    sudo tc class add dev eno1 parent 1: classid 1:1 htb rate 1000Mbps
    sudo tc class add dev eno1 parent 1:1 classid 1:11 htb rate $2
    sudo tc qdisc add dev eno1 parent 1:11 handle 10: netem delay 1ms
fi

dstat -n > bandwidth_$1_$2_$3_$4.csv &
python -m $3.main --world-size 3 --rank $1 $arg_ps --dist-url 'tcp://node0:8088' --quantize-nbits $4;

if [[ $2 != "0" ]]
then
    sudo tc qdisc del dev eno1 parent 1:11 handle 10: netem delay 1ms
    sudo tc class del dev eno1 parent 1:1 classid 1:11 htb rate $2
    sudo tc class del dev eno1 parent 1: classid 1:1 htb rate 1000Mbps
    sudo tc qdisc del dev eno1 handle 1: root htb default 11
fi

end_time=$(date +%s)
diff=$(( $end_time - $start_time ))
echo "It took $diff seconds"
terminate_cluster


