# cs744_project

## aeasgd usage:
```
python main.py --world-size 3 --rank 0 --flag --dist-url 'tcp://node0:8088' --quantize-nbits 8
python main.py --world-size 3 --rank 1 --no-flag --dist-url 'tcp://node0:8088' --quantize-nbits 8
python main.py --world-size 3 --rank 2 --no-flag --dist-url 'tcp://node0:8088' --quantize-nbits 8
```


## downpour sgd usage:
```
python main.py --world-size 3 --rank 0 --dist-url 'tcp://node0:8088' --server
python main.py --world-size 3 --rank 1 --dist-url 'tcp://node0:8088'
python main.py --world-size 3 --rank 2 --dist-url 'tcp://node0:8088'
```

## bandwidth.sh
```
bandwidth.sh {rank} {bandwidth(xMbit)} {downpoursgd2|easgd}
```

## latency.sh
```
latency.sh {rank} {latency(xms)} {downpoursgd2|easgd}
```
