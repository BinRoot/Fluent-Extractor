import numpy as np
import csv
import fluent_data_loader as fdl
from matplotlib import pyplot as plt
import minimaxent2


def get_utility_functions():
    #-3604.10421571
    # params1 = [2.12870472573, 0.611436743941, 0.118279428803, 2.34514902859, -0.0675109618653, 0.514521927042,
    #            0.298442475606, 0.323150370342, 2.00063660343]
    # params2 = [-0.0469531986593, 1.45331307612, -0.0731780573981, -0.106077559418, -0.0934629416504, 2.44256811397,
    #            2.18029163344, 1.85513772598, -0.00143000117458]
    # params3 = [2.12667524384, 2.18087287459, -0.224315016713, -0.102802055005, 0.474073009513, 0.503604557759,
    #            2.35190625203, 0.187351120688, 2.28758438518]
    # params4 = [2.36471633917, 1.61144152084, 0.297752055401, -0.244041336599, -0.380482225665, 2.27193867064,
    #            0.290179714562, 0.316140313564, 1.834683141]
    # params5 = [0.140766497404, 1.91039496936, 2.03353553812, -0.294041729361, -0.0529490158517, -0.160796266995,
    #            0.0541414126792, 2.13662903381, 0.0868740607558]
    # params6 = [1.66838628174, 2.42649517551, 2.25671313158, 2.25423621707, 2.4312868156, -0.0339174819718,
    #            2.12514882912, 0.0313072940236, 0.14636004983]
    # params7 = [1.62767326112, 1.85430210937, -0.0208675736734, 2.01695092291, -0.227888815281, -0.467934305235,
    #            2.20340719519, -0.243136155857, 2.12117941891]
    # params8 = [-0.258050934101, -0.108757313159, 0.16515754259, 0.075640164409, 2.13453904518, -0.000451435315659,
    #            2.5178627614, 1.87325701362, 1.64422038652]
    # params9 = [2.38819167895, 0.247793218312, 0.307999523906, -0.0383400411635, -0.429559435659, 1.73858986592,
    #            -0.127767567405, 2.10359807759, 2.42818474669]
    # params10 = [0.0336007334387, -0.122622231758, 1.78526326978, 1.84149724522, 2.09473143067, 1.87234982812,
    #             -0.00328780063638, 2.40417205394, 0.306186437309]
    # params11 = [2.28487651362, 2.32528707017, -0.0609635362322, 0.112020172182, 1.48264759141, 0.553797485864,
    #             1.89770686254, 2.59909248939, 2.14438677488]
    # params12 = [1.99492775993, 2.26602039769, 2.02698875738, 1.80007034383, 0.0243085080979, 2.20156390532,
    #             0.0921749565083, 0.122217669081, 1.80675452187]

    # -6221.14995225
    params1 = [2.40530044451, 1.97014509839, 2.20410538899, -0.299328356219, -0.578599998617, 1.96866924336]
    params2 = [2.45341008951, -0.716266511933, 1.95689332183, 2.76569985087, 2.04682819669, 1.39716248224]
    params3 = [2.05346945067, -0.198122052741, -0.32971340053, 2.50710511367, 2.51821341113, 1.67494090229]
    params4 = [-0.147365883246, -0.0126080408743, -0.189673747214, 2.58838119158, 1.98623609101, -0.241578820251]
    params5 = [2.09003307542, 2.45959704994, 2.12538115746, -0.502867668125, -0.0687230087765, 1.82952645496]
    params6 = [2.7782949047, 2.49954479527, 1.97109410797, -0.401661822405, 0.246164535168, 0.0058710092325]
    params7 = [2.10293070061, 2.08693400471, 2.15239691062, -0.522471191597, -0.187110342039, -0.354239965177]
    params8 = [-0.264152716185, 2.74253560739, 1.74126464014, -0.0106108397623, -0.313887381243, -0.137547573574]
    params9 = [0.140411968859, -0.546846100525, 2.8455380799, 0.290567039187, 0.519434199477, 0.0596888732337]
    params10 = [1.67585130044, 0.119047559563, -0.645616857482, 2.49900861188, -0.247665619748, 0.0528539237629]
    params11 = [0.169197207506, -0.596076512181, -0.357346804143, 2.40223627571, 0.212008736511, 1.70759671742]
    params12 = [-0.0475295265654, -0.108096863406, -0.673335727139, 2.27445378416, 0.482637090771, -0.0773386186623]

    all_params = [params1, params2, params3, params4, params5, params6,
                  params7, params8, params9, params10, params11, params12]
    utility_functions = []
    for params in all_params:
        pwf = minimaxent2.PiecewiseFunction(n_pieces=len(params1) - 1, joints_y=params)
        utility_functions.append(pwf)
    return utility_functions


if __name__ == '__main__':
    survey_filename = 'survey.csv'

    survey_results = []

    with open(survey_filename, 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        csvfile.readline()
        for row in csvreader:
            result = [-1 if field[-1] == 'A' else 1 for field in row[1:]]
            survey_results.append(result)

    human_preferences = np.mean(survey_results, axis=0)
    print(human_preferences)

    q1 = '20:0', '20:49'
    q2 = '22:0', '22:25'
    q3 = '25:25', '20:49'
    q4 = '28:19', '35:15'
    q5 = ('5:0', '5:40'), ('6:0', '6:28')
    q6 = ('16:22', '16:29'), ('25:25', '25:32')
    q7 = ('13:12', '13:20'), ('17:13', '17:22')


    loader = fdl.DataLoader()
    utility_functions = get_utility_functions()

    robot_preferences = []

    f1a = loader.get_fluents_from_meta(q1[0])
    f1b = loader.get_fluents_from_meta(q1[1])
    u1a = minimaxent2.utility(f1a, utility_functions)
    u1b = minimaxent2.utility(f1b, utility_functions)
    robot_preferences.append(u1a - u1b)

    f2a = loader.get_fluents_from_meta(q2[0])
    f2b = loader.get_fluents_from_meta(q2[1])
    u2a = minimaxent2.utility(f2a, utility_functions)
    u2b = minimaxent2.utility(f2b, utility_functions)
    robot_preferences.append(u2a - u2b)

    f3a = loader.get_fluents_from_meta(q3[0])
    f3b = loader.get_fluents_from_meta(q3[1])
    u3a = minimaxent2.utility(f3a, utility_functions)
    u3b = minimaxent2.utility(f3b, utility_functions)
    robot_preferences.append(u3a - u3b)

    f4a = loader.get_fluents_from_meta(q4[0])
    f4b = loader.get_fluents_from_meta(q4[1])
    u4a = minimaxent2.utility(f4a, utility_functions)
    u4b = minimaxent2.utility(f4b, utility_functions)
    robot_preferences.append(u4a - u4b)

    f5a = loader.get_fluents_from_meta(q5[0][0]), loader.get_fluents_from_meta(q5[0][1])
    f5b = loader.get_fluents_from_meta(q5[1][0]), loader.get_fluents_from_meta(q5[1][1])
    u5a = minimaxent2.utility(f5a[0], utility_functions), minimaxent2.utility(f5a[1], utility_functions)
    u5b = minimaxent2.utility(f5b[0], utility_functions), minimaxent2.utility(f5b[1], utility_functions)
    robot_preferences.append((u5b[0] - u5b[1]) - (u5a[0] - u5a[1]))

    f6a = loader.get_fluents_from_meta(q6[0][0]), loader.get_fluents_from_meta(q6[0][1])
    f6b = loader.get_fluents_from_meta(q6[1][0]), loader.get_fluents_from_meta(q6[1][1])
    u6a = minimaxent2.utility(f6a[0], utility_functions), minimaxent2.utility(f6a[1], utility_functions)
    u6b = minimaxent2.utility(f6b[0], utility_functions), minimaxent2.utility(f6b[1], utility_functions)
    robot_preferences.append((u6b[0] - u6b[1]) - (u6a[0] - u6a[1]))

    f7a = loader.get_fluents_from_meta(q7[0][0]), loader.get_fluents_from_meta(q7[0][1])
    f7b = loader.get_fluents_from_meta(q7[1][0]), loader.get_fluents_from_meta(q7[1][1])
    u7a = minimaxent2.utility(f7a[0], utility_functions), minimaxent2.utility(f7a[1], utility_functions)
    u7b = minimaxent2.utility(f7b[0], utility_functions), minimaxent2.utility(f7b[1], utility_functions)
    robot_preferences.append((u7b[0] - u7b[1]) - (u7a[0] - u7a[1]))

    robot_preferences = np.array(robot_preferences)
    robot_preferences[0:4] /= np.max(robot_preferences[0:4])
    robot_preferences[4:] /= (max(np.max(robot_preferences[4:]), -np.min(robot_preferences[4:])) / 0.618)

    plt.figure()
    bar_index = np.arange(len(human_preferences))
    bar_width = 0.35
    rects_1 = plt.barh(bar_index - bar_width/2,
                      np.fliplr([human_preferences])[0],
                      bar_width,
                      color='b')
    rects_2 = plt.barh(bar_index + bar_width/2,
                      np.fliplr([robot_preferences])[0],
                      bar_width,
                      color='r')
    plt.yticks(bar_index + bar_width / 2, ('G', 'F', 'E', 'D', 'C', 'B', 'A'))
    # plt.xticks([])
    plt.show()



