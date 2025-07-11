MyoChallenge-2025 Documentations
#############################################


* :ref:`challenge25_table_tennis_rally`
* :ref:`challenge25_soccer_shootout`



.. _challenge25_table_tennis_rally:

Table Tennis Rally
--------------------------------------------------------------

The agent must hit a pingpong ball such that the ball lands on the opponent's side using a paddle. This task requires coordination of a 
'myoArm' model and a 'myoTorso' model as to allow the agent to accurately hit the pingpong ball without missing and allowing enough force 
so that the ball reaches within the dimensions of the opponent's side. 


.. image:: images/MyoChallenge25TableTennis.png
    :width: 450
    :align: center



Objective
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To develop a general policy to  engage in a high-speed rally.
Move the ball from the agent's side to the opposite side by hitting the ball with a paddle.



Action Space
^^^^^^^^^^^^^^^^^^^^^^^^
The action space includes three major parts, the :ref:`myoArm`, consisting of 63 muscles, the :ref:`myoTorso`, consisting of 210 muscles 
and two position actuators for pelvis translation in the x,y plane. 


Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^


.. temporary change backup
.. +-----------------------------------------+-----------------------------+-----------------+
.. | **Description**                         |      **Component**          |   **Count**     |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Pelvis Position                         | pelvis_pos                  |  (3)            |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Joint Positions                         | body_qpos                   |  (86)           |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Joint Velocities                        | body_vel                    |  (86)           | 
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Ball Position                           | ball_pos                    |  (3)            |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Ball Velocity                           | ball_vel                    |  (3)            |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Paddle Position                         | paddle_pos                  |  (3)            |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Paddle Velocity                         | paddle_vel                  |  (3)            |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Paddle Reaching Error                   | reach_err                   |  (3)            |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Muscle Activations                      | muscle_activations          |  (273)          |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Touching Information                    | touching_info               |  (6)            |
.. +-----------------------------------------+-----------------------------+-----------------+



+-----------------------------------+--------------------+-----------+
| **Description**                   | **Component**      | **Count** |
+-----------------------------------+--------------------+-----------+
| Pelvis Position                   | pelvis_pos         | 3         |
+-----------------------------------+--------------------+-----------+
| Joint Positions                   | body_qpos          | 86        |
+-----------------------------------+--------------------+-----------+
| Joint Velocities                  | body_vel           | 86        |
+-----------------------------------+--------------------+-----------+
| Ball Position                     | ball_pos           | 3         |
+-----------------------------------+--------------------+-----------+
| Ball Velocity                     | ball_vel           | 3         |
+-----------------------------------+--------------------+-----------+
| Paddle Position                   | paddle_pos         | 3         |
+-----------------------------------+--------------------+-----------+
| Paddle Velocity                   | paddle_vel         | 3         |
+-----------------------------------+--------------------+-----------+
| Paddle Reaching Error (see below) | reach_err          | 3         |
+-----------------------------------+--------------------+-----------+
| Muscle Activations                | muscle_activations | 273       |
+-----------------------------------+--------------------+-----------+
| Touching Information (see below)  | touching_info      | 6         |
+-----------------------------------+--------------------+-----------+





**Description of observations**

    - The paddle reaching error measures the distance between the MPL and the object
    - The touching information indicates the pingpong ball's contact with various objects in the environment:
        - Paddle: Whether the ball is in contact with the paddle.
        - Own: Whether the ball is in contact with the agent.
        - Opponent: Whether the ball is in contact with an opponent agent.
        - Ground: Whether the ball is in contact with the ground.
        - Net: Whether the ball is in contact with the net.
        - Env: Whether the ball is in contact with any part of the environment. 



**Ping Pong Object Properties**

+-----------------------+----------------------------------+
| **Object**            | **Properties**                   |
+-----------------------+----------------------------------+
| Table Top (Total)     | 2.74 × 1.52 × 1.59 m³            |
+-----------------------+----------------------------------+
| Table Top (Each Side) | 1.37 × 1.52 × 1.59 m³            |
+-----------------------+----------------------------------+
| Net                   | 0.01 × 1.825 × 0.305 m³          |
+-----------------------+----------------------------------+
| Paddle Handle         | Radius: 1.6 cm, Height: 5.1 cm   |
+-----------------------+----------------------------------+
| Paddle Face           | Radius: 9.3 cm, Height: 2 cm     |
+-----------------------+----------------------------------+
| Paddle Mass           | 150 g                            |
+-----------------------+----------------------------------+
| Ball Radius           | 2 cm                             |
+-----------------------+----------------------------------+
| Ball Mass             | 2.7 g                            |
+-----------------------+----------------------------------+
| Ball Inertia          | 7.2×10⁻⁷ kg·m²                   |
+-----------------------+----------------------------------+





Starting Criteria: Phase 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- The ball starts at the same position with the same speed
- The agent has the same starting position
- The paddle initially starts in the grasping position with the hand,
  but is not connected.


Success Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^

- The ball is hit by the paddle once and only once
- The ball does not have contact with the agent's side of the table
- The ball hits the opponent's side of the table


Ranking Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Success rate of hitting the ball (successful_attempts / total_attempts)
2. Effort: based on muscle activation energy



.. _challenge25_soccer_shootout:

Soccer Shootout
--------------------------------------------------------------

The locomotion task focuses on goal-scoring using dynamic muscular control. 
The agent must kick a soccer ball, such that it enter's the goal net. This task requires coordination of a 'myoLeg' model and a 'myoTorso' model as to 
allow the agent to accurately hit the ball without missing and allowing enough force that the ball 
reaches within the confines of the net.


.. image:: images/MyoChallenge25Soccer.png
    :width: 450
    :align: center



Objective
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To develop policies that allow for coordinated locomotion and kicking of a ball to score goals 
in a net with and without a goalkeeper.


Action Space
^^^^^^^^^^^^^^^^^^^^^^^^
The action space includes two major parts, the :ref:`myoLeg`, consiting of 80 leg muscles, and the :ref:`myoTorso`, consisting of 210 lumabr muscles. 


Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^


.. temporary change backup
.. +-----------------------------------------+-----------------------------+-----------------+
.. | **Description**                         |      **Component**          |   **Count**     |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Ball Position                           | ball_pos                    | (3)             |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | 4 Position Coords (bounding goal area)  | goal_bounds                 | (12)            | 
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Muscles Activations                     | act                         | (290)           |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Joint Angles                            | internal_qpos               | (46)            |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Joint Velocities                        | internal_qvel               | (46)            |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Foot Position (Right)                   | r_toe_pos                   | (3)             |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Ball Contact Forces with Foot           | l_toe_pos                   | (3)             |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Body COM in world frame                 | model_root_pos              | (7)             |
.. +-----------------------------------------+-----------------------------+-----------------+
.. | Body COM vel in world frame             | model_root_vel              | (6)             |
.. +-----------------------------------------+-----------------------------+-----------------+


+----------------------------------------+----------------+-----------+
| **Description**                        | **Component**  | **Count** |
+----------------------------------------+----------------+-----------+
| Ball Position                          | ball_pos       | 3         |
+----------------------------------------+----------------+-----------+
| 4 Position Coords (bounding goal area) | goal_bounds    | 12        |
+----------------------------------------+----------------+-----------+
| Muscles Activations                    | act            | 290       |
+----------------------------------------+----------------+-----------+
| Joint Angles                           | internal_qpos  | 46        |
+----------------------------------------+----------------+-----------+
| Joint Velocities.                      | internal_qvel  | 46        |
+----------------------------------------+----------------+-----------+
| Foot Position (Right)                  | r_toe_pos      | 3         |
+----------------------------------------+----------------+-----------+
| Foot Position (Left)                   | l_toe_pos      | 3         |
+----------------------------------------+----------------+-----------+
| Body COM in world frame                | model_root_pos | 7         |
+----------------------------------------+----------------+-----------+
| Body COM vel in world frame            | model_root_vel | 6         |
+----------------------------------------+----------------+-----------+




**Soccer Object Properties**


+-----------------------+----------------------------------+
| **Object**            | **Properties**                   |
+-----------------------+----------------------------------+
| Soccer Net            | Width: 7.32 m, Height: 2.50 m    |
+-----------------------+----------------------------------+
| Ball Radius           | 0.117m                           |
+-----------------------+----------------------------------+
| Ball Mass             | 450g                             |
+-----------------------+----------------------------------+




Starting Criteria: Phase 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- The human model is placed in a fixed starting location, directly in front of the ball, which is also placed in a fixed starting location. 

.. Starting Criteria: Phase 2 (upcoming)
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. - The ball is placed in a fixed starting location. The human model is placed at random locations within a fixed radius of the ball, 
  and as before always placed in front of the ball. As well, a goalkeeper model is present, following a public policy with static and random movement. 


Success Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^

1. The soccer ball is fully within the confines of the net.
2. The agent scores within 20 seconds.


Ranking Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Success rate of scoring goals (goals_scored / total_attemps)
2. Effort: based on muscle activation energy

