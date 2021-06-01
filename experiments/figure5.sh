for ((j=100; j<=500; j+=100)); do
for ((i=1; i<=10;i+=1)); do
python3 -m scripts.train --env MiniGrid-Maze-v0 --algo a2c --model a2c-maze-$i --episodes $j --procs 10
python3 -m scripts.train --env MiniGrid-Maze-v0 --algo ppo --model ppo-maze-$i --episodes $j --procs 10
python3 -m scripts.train --env MiniGrid-Maze-Intermediate-v0 --algo a2c --model a2c-intermaze10000-$i --episodes $j --procs 10
python3 -m scripts.train --env MiniGrid-Maze-Intermediate-v0 --algo ppo --model ppo-intermaze-$i --episodes $j --procs 10
python3 -m scripts.train --env MiniGrid-ThreeDoor-v0 --algo a2c --model a2c-3door-$i --episodes $j --procs 10
python3 -m scripts.train --env MiniGrid-ThreeDoor-v0 --algo ppo --model ppo-3door-$i --episodes $j --procs 10
python3 -m scripts.train --env MiniGrid-SparseThreeDoor-v0 --algo a2c --model a2c-s3door-$i --episodes $j --procs 10
python3 -m scripts.train --env MiniGrid-SparseThreeDoor-v0 --algo ppo --model ppo-s3door-$i --episodes $j --procs 10
done
for ((i=1; i<=10;i+=1)); do
python3 -m scripts.train --env MiniGrid-Maze-v0 --algo a2c --model a2c-maze-$i --episodes 10 --procs 10
python3 -m scripts.train --env MiniGrid-Maze-v0 --algo ppo --model ppo-maze-$i --episodes 10 --procs 10
python3 -m scripts.train --env MiniGrid-Maze-Intermediate-v0 --algo a2c --model a2c-intermaze-$i --episodes 10 --procs 10
python3 -m scripts.train --env MiniGrid-Maze-Intermediate-v0 --algo ppo --model ppo-intermaze-$i --episodes 10 --procs 10
python3 -m scripts.evaluate --env MiniGrid-ThreeDoor-v0 --episodes 10 --algo a2c --model a2c-3door-$i --procs 10
python3 -m scripts.evaluate --env MiniGrid-ThreeDoor-v0 --episodes 10 --algo ppo --model ppo-3door-$i --procs 10
python3 -m scripts.evaluate --env MiniGrid-SparseThreeDoor-v0 --episodes 10 --algo a2c --model a2c-s3door-$i --procs 10
python3 -m scripts.evaluate --env MiniGrid-SparseThreeDoor-v0 --episodes 10 --algo ppo --model ppo-s3door-$i --procs 10
done
done

for ((j=2000; j<=10000; j+=2000)); do
for ((i=1; i<=10;i+=1)); do
python3 -m scripts.train --env MiniGrid-Maze-v0 --algo dqn --model dqn-maze-$i --episodes $j --procs 1
python3 -m scripts.train --env MiniGrid-Maze-Intermediate-v0 --algo dqn --model dqn-intermaze-$i --episodes $j --procs 1
python3 -m scripts.train --env MiniGrid-ThreeDoor-v0 --algo dqn --model dqn-3door-$i --episodes $j --procs 1
python3 -m scripts.train --env MiniGrid-SparseThreeDoor-v0 --algo dqn --model dqn-s3door-$i --episodes $j --procs 1
done
for ((i=1; i<=10;i+=1)); do
python3 -m scripts.train --env MiniGrid-Maze-v0 --episodes 10 --algo dqn --model dqn-maze-$i --procs 10
python3 -m scripts.train --env MiniGrid-Maze-Intermediatev0 --episodes 10 --algo dqn --model dqn-intermaze-$i --procs 10
python3 -m scripts.evaluate --env MiniGrid-ThreeDoor-v0 --episodes 10 --algo dqn --model dqn-3door-$i --procs 10
python3 -m scripts.evaluate --env MiniGrid-SparseThreeDoor-v0 --episodes 10 --algo dqn --model dqn-s3door-$i --procs 10
done
done
