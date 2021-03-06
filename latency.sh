# usage run.sh {rank} {latency(xms)} {downpoursgd2|easgd} {quantize-nbits}

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

sudo tc qdisc add dev eno1 root netem delay $2 

dstat -n > network_$3_$2.csv &
cd ../
python -m cs744_project.$3.main --world-size 3 --rank $1 $arg_ps --dist-url 'tcp://node0:8088' --quantize-nbits $4;
cd ./cs744_project
sudo tc qdisc del dev eno1 root netem delay $2 



end_time=$(date +%s)
diff=$(( $end_time - $start_time ))
echo "It took $diff seconds"
terminate_cluster



