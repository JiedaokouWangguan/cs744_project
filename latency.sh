ssh -p 22 yfsong@c220g5-110914.wisc.cloudlab.us "sudo tc qdisc add dev eno1 root netem delay 100ms ; cd ./cs744_project/easgd;dstat -n --output network.csv & python main.py --world-size 3 --rank 0 --flag --dist-url 'tcp://node0:8088' --quantize-nbits 8 > result.txt ; sudo tc qdisc del dev eno1 root netem delay 100ms"
ssh -p 22 yfsong@c220g5-110916.wisc.cloudlab.us "sudo tc qdisc add dev eno1 root netem delay 100ms ; cd ./cs744_project/easgd;dstat -n --output network.csv & python main.py --world-size 3 --rank 1 --no-flag --dist-url 'tcp://node0:8088' --quantize-nbits 8 > result.txt ; sudo tc qdisc del dev eno1 root netem delay 100ms"
ssh -p 22 yfsong@c220g5-110915.wisc.cloudlab.us "sudo tc qdisc add dev eno1 root netem delay 100ms ; cd ./cs744_project/easgd;dstat -n --output network.csv & python main.py --world-size 3 --rank 1 --no-flag --dist-url 'tcp://node0:8088' --quantize-nbits 8 > result.txt ; sudo tc qdisc del dev eno1 root netem delay 100ms"
