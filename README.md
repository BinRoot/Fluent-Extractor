# Setup

1. Follow steps 1-3 http://sdk.rethinkrobotics.com/wiki/Workstation_Setup

2. Put this `fluent_extractor` code in `ros_ws/src/`. Then run the following:

    $ catkin_make

3. Edit the `config.json` file located `ros_ws/src/fluent_extractor`.

3. Run the `vision_buffer`

    rosrun fluent_extractor vision_buffer src/fluent_extractor/config.json
