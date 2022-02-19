<!-- =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= -->

# MyoSuite
MyoSuite consists of three models -- `Finger`, `Elbow` and `Hand`. Using these models we design a rich collection of tasks ranging from simple reaching movements to contact-rich movements like pen-twirling and baoding balls. The figure below provides an overview of the tasks included the MyoSuite.
<img width="1240" alt="TasksALL" src="https://user-images.githubusercontent.com/23240128/134825260-0de32d74-e096-4ea5-906d-26302fade35f.png">
Furthermore, all MyoSuite tasks have multiple difficulty variations and non-stationary variations --


## 1. Task Descriptions and Difficulty Variations
<details>
  <summary>Click to read more about task descriptions and their difficulty variations</summary>

| Tasks  | Details +  Difficulty Variations|
|-----------|--------------------|
|<img src="https://user-images.githubusercontent.com/23240128/134833319-a68c2768-5a15-4aed-ac5c-62c80b631da9.png" width="800">| **Finger Joint Pose -** <br><br> Objective--: Strike a joint pose <br> Easy variant: Move to a fixed specified joint pose (_myoFingerPoseFixed-v0_) <br> Hard variant: Move to randomly selected joint poses (_myoFingerPoseRandom-v0_) <br> More variant: Random resets, no resets, dense rewards, sparse rewards|
|<img src="https://user-images.githubusercontent.com/23240128/134833325-d5d21b79-816f-4341-8896-5f81a23b818b.png" width="800">|**Finger Tip Reach -** <br><br>Objective--: Reach using figner tips <br> Easy variant: Reach to a fixed location (_myoFingerReachFixed-v0_) <br> Hard variant: Reach to random locations (_myoFingerReachRandom-v0_) <br> More variant: Random resets, no resets, dense rewards, sparse rewards|
|<img src="https://user-images.githubusercontent.com/23240128/134833335-46386181-e394-4b38-a531-28af3856d8a7.png" width="800">|**Elbow Joint Pose -** An elbow model with 6 muscles (3 flexors and 3 extensors) was simplified to have only elbow rotations. Although it is not a highly physiologically accurate model it can be a very simple model for troubleshooting initial control schemes. <br><br> Objective--: Move elbow to a specified pose. <br>Easy variant: Move to specified fixed joint pose (_myoElbowPose1D6MFixed-v0_) <br> Hard variant: Move to random joint poses (_myoElbowPose1D6MRandom-v0_) <br> More variant: Random resets, no resets, dense rewards, sparse rewards|
|<img src="https://user-images.githubusercontent.com/23240128/134833345-5ea2eff7-1a29-4c40-a1e4-23b7e9e872b7.png" width="800">|**Hand Joints Pose -** Drive full forearm-wrist-hand to make joint poses. In addition to making co-ordinated movements, avoiding self collisions poses additional challenges in solving this task. <br><br> Objective--: Strike a hand pose <br>Easy variant: Move to a fixed joint pose (_myoHandPoseFixed-v0_) <br> Hard variant: Move to a randomly selected joint pose (_myoHandPoseRandom-v0_) <br> More variant: Random resets, no resets, dense rewards, sparse rewards|
|<img src="https://user-images.githubusercontent.com/23240128/134833356-bf51dae8-3488-477f-9d3e-646ccf056bf1.png" width="800">|**Hand Tips Reach -** Make reaching movements using full forearm-wrist-hand. In addition to making co-ordinated movements, avoiding self collisions poses additional challenges in solving this task. <br><br> Objective--: Reach using finger tips <br>Easy variant: Reach fixed positions using finger tips (_myoHandReachFixed-v0_) <br> Hard variant: Reach random positions using finger tips (_myoHandReachRandom-v0_) <br> More variant: Random resets, no resets, dense rewards, sparse rewards|
|<img src="https://user-images.githubusercontent.com/23240128/134833373-feef95d1-1768-47ac-9f68-602dc21fb0d5.png" width="800">|**Hand Key Turn -** In this task a simplified model of the myo-hand with only thumb and index muscles was used. This model consisted of 20 muscles (all the forearm and hand muscle with the exclusion of the muscle to control middle, ring and little fingers). <br><br> Objective--: Coordinate finger movements to rotate a key <br>Easy variant: Achieve half rotation of the key (_myoHandKeyTurnFixed-v0_) <br> Hard variant: Achieve full rotation + random initial configuration (_myoHandKeyTurnRandom-v0_) <br> More variant: Random resets, no resets, dense rewards, sparse rewards |
|<img src="https://user-images.githubusercontent.com/23240128/134833381-a31dce47-6c67-4911-9525-7c13f63cade6.png" width="800">|**Hand Object Hold -**  A full forearm-wrist-hand moves an object in the hand to a given orientation without dropping. The complexity of this task is due to the intermittent contacts between the object and multiple fingers needing co-ordination to stabilize the object. <br><br> Objective--: Reposition an object to reach a given target without dropping it. <br>Easy variant: Reposition to a fixed position (_myoHandObjHoldFixed-v0_) <br> Hard variant: Reposition a random object to random positions (_myoHandObjHoldRandom-v0_) <br> More variant: sparse rewards, dense rewards, random resets, reset free|
|<img src="https://user-images.githubusercontent.com/23240128/134833391-962c5076-9215-4170-b02f-a41eb1092b37.png" width="800">|**Hand Pen Twirl -** A full forearm-wrist-hand rotate a pen in the hand to a given orientation without dropping. The complexity of this task is due to the intermittent contacts between the object and multiple fingers while trying to stabilize the object. <br><br> Objective--: Rotate the object to reach a given orientation (indicated by the green object in the scene) without dropping it. <br>Easy variant: Rotate to fixed orientation (_myoHandPenTwirlFixed-v0_) <br> Hard variant: Rotate of random orientation (_myoHandPenTwirlRandom-v0_) <br> More variant: sparse rewards, dense rewards, random resets, reset free|
|<img src="https://user-images.githubusercontent.com/23240128/134833398-0a9318ec-a980-4cf3-b702-2d1939f5979e.png" width="800">|**Hand Baoding Balls -** A baoding ball task involving simultaneous rotation of two free-floating spheres over the palm. This task requires both dexterity and coordination. <br><br> Objective--: Achieve relative rotation of the balls around each other without dropping them. <br>Easy variant: Swap the position of the balls (_myoHandBaodingFixed-v1_) <br> Hard variant: Achieve contineous rotations (_myoHandBaodingRandom-v1_) <br> More variant: Sparse rewards, 3 different dense reward options to choose from|
</details>

## 2. Non-Stationarity variations
<details>
  <summary>Click to read more about task's non-stationarity variations</summary>

|                    | **Environment**                 | **Difficulty**  | **Sarcopenia** | **Fatigue** | **Tendon-transfer** |
|--------------------|---------------------------------|----------------|------------------|---------|-----------------|
| Finger Joint Pose  | _myoFingerPoseFixed-v0_          | Easy       | √          | √       |                 |
| Finger Joint Pose  | _myoFingerPoseRandom-v0_         | Hard       | √          | √       |                 |
| Finger Tip Reach   | _myoFingerReachFixed-v0_         | Easy       | √          | √       |                 |
| Finger Tip Reach   | _myoFingerReachRandom-v0_        | Hard       | √          | √       |                 |
| Elbow Joint Pose   | _myoElbowPose1D6MRandom-v0_      | Hard       | √          | √       |                 |
| Elbow Joint Pose   | _myoElbowPose1D6MRandom-v0_      | Hard       | √          | √       |                 |
| Hand Joints Pose   | _myoHandPoseFixed-v0_            | Easy       | √          | √       | √               |
| Hand Joints Pose   | _myoHandPoseRandom-v0_           | Hard       | √          | √       | √               |
| Hand Tips Reach    | _myoHandReachFixed-v0_           | Easy       | √          | √       | √               |
| Hand Tips Reach    | _myoHandReachRandom-v0_          | Hard       | √          | √       | √               |
| Hand Key Turn      | _myoHandKeyTurnFixed-v0_         | Easy       | √          | √       | √               |
| Hand Key Turn      | _myoHandKeyTurnRandom-v0_        | Hard       | √          | √       | √               |
| Hand Object Hold   | _myoHandObjHoldFixed-v0_         | Easy       | √          | √       | √               |
| Hand Object Hold   | _myoHandObjHoldRandom-v0_        | Hard       | √          | √       | √               |
| Hand Pen Twirl     | _myoHandPenTwirlFixed-v0_        | Easy       | √          | √       | √               |
| Hand Pen Twirl     | _myoHandPenTwirlRandom-v0_       | Hard       | √          | √       | √               |
| Hand Baoding Balls | _myoHandBaodingFixed-v1_         | Easy       | √          | √       | √               |
| Hand Baoding Balls | _myoHandBaodingRandom-v1_        | Hard       | √          | √       | √               |