for ((i=1; j<=10;i+=1)); do
python3 -m scripts.train --env MiniGrid-FourDoor-v0 --algo dqn --model dqn-4Door-1000000-$i --frames 1000000 --procs 1 --batch-size 300 --max-memory 1500 --seed $i --update-interval 20
done
