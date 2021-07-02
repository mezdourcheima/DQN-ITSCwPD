import time

# """CHANGE CLOCK SCHEDULE INTERVAL HERE""" ############################################################################
DT = 0.
########################################################################################################################


class SumoView:
    def __init__(self, name, env):
        self.name = name
        self.env = env

        self.setup()

        # """CHANGE VIEW SETUP HERE""" #################################################################################
        ################################################################################################################

    def get_play_action(self):
        play_action = 0

        # """CHANGE GET PLAY ACTION HERE""" ############################################################################
        ################################################################################################################

        return play_action

    def on_draw(self):
        # """CHANGE VIEW LOOP HERE""" ##################################################################################
        pass
        ################################################################################################################

    def clear(self):
        # """CHANGE CLEAR VIEW HERE""" #################################################################################
        pass
        ################################################################################################################

    def setup(self):
        raise NotImplementedError

    def loop(self):
        raise NotImplementedError

    def run(self):
        while True:
            self.clear()
            self.loop()
            self.on_draw()
            time.sleep(DT)
