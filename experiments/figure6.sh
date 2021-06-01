for ((j=500; j<=2500; j+=500)); do
for ((i=1; i<=10;i+=1)); do
python3 -m scripts.train --env MiniGrid-FourDoor-v0 --algo a2c --model a2c-4Door-1k-$i --episodes $j --procs 10
python3 -m scripts.train --env MiniGrid-FourDoor-v0 --algo ppo --model ppo-4Door-1k-$i --episodes $j --procs 10
python3 -m scripts.train --env MiniGrid-10FourDoor-v0 --algo a2c --model a2c-4Door-10-$i --episodes $j --procs 10
python3 -m scripts.train --env MiniGrid-10FourDoor-v0 --algo ppo --model ppo-4Door-10-$i --episodes $j --procs 10
done
for ((i=1; i<=10;i+=1)); do
python3 -m scripts.evaluate --env MiniGrid-FourDoor-v0 --episodes 10 --algo a2c --model a2c-4Door-1k-$i --procs 10
python3 -m scripts.evaluate --env MiniGrid-FourDoor-v0 --episodes 10 --algo ppo --model ppo-4Door-1k-$i --procs 10
python3 -m scripts.evaluate --env MiniGrid-10FourDoor-v0 --episodes 10 --algo a2c --model a2c-4Door-10-$i --procs 10
python3 -m scripts.evaluate --env MiniGrid-10FourDoor-v0 --episodes 10 --algo ppo --model ppo-4Door-10-$i --procs 10
done
done

for ((j=2000; j<=10000; j+=2000)); do
for ((i=1; i<=10;i+=1)); do
python3 -m scripts.train --env MiniGrid-FourDoor-v0 --algo dqn --model dqn-4Door-1k-$i --episodes $j --procs 10
python3 -m scripts.train --env MiniGrid-10FourDoor-v0 --algo dqn --model dqn-4Door-10-$i --episodes $j --procs 10
done
for ((i=1; i<=10;i+=1)); do
python3 -m scripts.evaluate --env MiniGrid-FourDoor-v0 --episodes 10 --algo dqn --model dqn-4Door-1k-$i --procs 10
python3 -m scripts.evaluate --env MiniGrid-10FourDoor-v0 --episodes 10 --algo dqn --model dqn-4Door-10-$i --procs 10
done
done
