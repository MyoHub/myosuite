MyoChallenge-2025 Documentations
#############################################


* :ref:`challenge25_table_tennis_manipulation`
* :ref:`challenge25_table_tennis_manipulation_locomotion`
* :ref:`challenge25_soccer`
* :ref:`challenge25_tutorial`



.. _challenge25_table_tennis_manipulation:

Prosthesis Table Tennis
--------------------------------------------------------------

Task Description: Using a paddle, the agent must hit a pingpong ball such that the ball lands on the opponent's side. This task requires coordination of a 
'myoArm' model and a 'myoTorso' model as to allow the agent to accurately hit the pingpong ball without missing and allowing enough force so that the ball 
reaches within the dimensions of the opponent's side. 


.. image:: images/MyoChallenge25TableTennis.png
    :width: 450
    :align: center



Objective
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To develop a general policy to  engage in a high-speed rally against an AI opponent. 
Move the object from the agent's side to the opponent's side by hitting the ball with a paddle.



Action Space
^^^^^^^^^^^^^^^^^^^^^^^^
The action space includes three major parts, the :ref:`myoArm`, consisting of 63 muscles, the `myoArm`, consisting of 210 muscles 
and two position actuators for pelvis translation in the x,y plane. 

The action for the prosthetic hand is controlled in terms of each joint angle. A normalisation is applied such that all joint angle in radiance can be 
actuated by a control value between  :math:`[-1, 1]`, with -1 and 1 representing the lower and upper bound of the range of motions.


Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^


.. temporary change backup
.. +-----------------------------------------+-----------------------------+-----------------+
.. | **Description**                         |      **Component**          |   **Count**     |
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
.. | Muscles Activations                     | muscle_activations          |  (273)          |
.. +-----------------------------------------+-----------------------------+-----------------+


+-----------------------------------------+-----------------------------+-----------------+
| **Description**                         |      **Component**          |     **Count**   |
+-----------------------------------------+-----------------------------+-----------------+
| Joint Positions                         | body_qpos                   | (86)            |
+-----------------------------------------+-----------------------------+-----------------+
| Joint Velocities                        | body_vel                    | (86)            | 
+-----------------------------------------+-----------------------------+-----------------+
| Ball Position                           | ball_pos                    | (3)             |
+-----------------------------------------+-----------------------------+-----------------+
| Ball Velocity                           | ball_vel                    | (3)             |
+-----------------------------------------+-----------------------------+-----------------+
| Paddle Position                         | paddle_pos                  | (3)             |
+-----------------------------------------+-----------------------------+-----------------+
| Paddle Velocity                         | paddle_vel                  | (3)             |
+-----------------------------------------+-----------------------------+-----------------+
| Paddle Reaching Error                   | reach_err                   | (3)             |
+-----------------------------------------+-----------------------------+-----------------+
| Muscle Activations                      | muscle_activations          | (273)           |
+-----------------------------------------+-----------------------------+-----------------+




**Description of observations**

    - Hand paddle error measures the distance between the MPL and the object

    - The pingpong ball has full 6 degrees of freedom.



**Object Properties**

Ping Pong Table:
- Table top:
    Total: 1.37 x 1.52 x 1.59 m^3
    Per side (agent/opponent): 0.685 x 0.76 x 0.795 m^3
- Net dimensions: 0.005 x 0.9125 x 0.1525 m^3

Paddle:
- Handle: radius = 0.016m, height = 0.051 m
- Face: radius = 0.093m, height = 0.020 m
- Mass: 100 g

Ball:
- Radius: 0.02m
- Mass: 2.7 g
- Inertia: 7.2e-7 kgm^2



Starting Criteria: Phase 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- The ball starts at the same position with the same speed
- The agent has the same starting position
- The paddle is fixed to the hand with the ball joint


Success Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^

- The ball is hit by the paddle once and only once
- The ball does not have contact with the agent's side of the table
- The ball hits the opponent's side of the table


Ranking Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^

1. Success rate (of hitting the ball) (successful_attempts / total_attempts)
2. Effort: based on muscle activation energy

