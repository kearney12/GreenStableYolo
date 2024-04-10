import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

# Specify the CSV file path

plot_points = True

csv_file = "GreenStableYolo_results_numgen50_popsize25.csv"

runs = []

average_HY_StableYolo = []
average_HY_GreenStableYolo = []
hypervolume_points_GreenStableYolo = []
hypervolume_points_StableYolo = []
inference_times = []

with open(csv_file, "r") as file:
    reader = csv.reader(file)
    for row in reader: # Process each row of the CSV file
        len_row = len(row)
        run = int(row[0])
        if run not in runs:
            runs.append(run)
        else:
            print('Run ', run)

            hypervolume_points = []
            for i in range(1, len_row):
                individual = row[i].split(", ")
                if len(individual) == 2:
                    image_quality = float(individual[0].replace("(", ""))
                    inference_time = float(individual[1].replace(")", ""))
                    print('Individual {}, image quality: {}, inference time (ms): {}'.format(i, image_quality, inference_time))
                    hypervolume_points.append([1-image_quality, inference_time])
                    inference_times.append(inference_time)
            # hypervolume_points_GreenStableYolo.append(hypervolume_points)


                    ref_point = np.array([1, 50000])
                    print('ref_point: ', ref_point)
                    from pymoo.indicators.hv import HV
                    indicator = HV(ref_point=ref_point)

                    hypervolume = indicator(np.array(hypervolume_points))
                    print('Hypervolume: ', hypervolume)
                    average_HY_GreenStableYolo.append(hypervolume)

                    if plot_points:
                        # plt.title('{}'.format(dir_data.replace('Data/', '').replace('.csv', '').replace('_AllNumeric', '')))
                        for point in hypervolume_points:
                            plt.scatter(point[0], point[1], color='blue', label='Ref_point')
                        plt.scatter(ref_point[0], ref_point[1], color='black', label='Ref_point')
                        plt.xlabel("1-image_quality")
                        plt.ylabel("inference_time", labelpad=5)
                        # plt.legend()


    csv_file = "StableYolo_results_numgen50_popsize25.csv"

    runs = []
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        for row in reader:  # Process each row of the CSV file
            len_row = len(row)
            run = int(row[0])
            if run not in runs:
                runs.append(run)
            else:
                print('Run ', run)

                hypervolume_points = []
                print(row)
                image_quality = float(row[1].replace("(", "").replace(",)", ""))
                inference_time = float(row[2])
                print('Individual {}, image quality: {}, inference time (ms): {}'.format(i, image_quality,
                                                                                             inference_time))
                hypervolume_points.append([1 - image_quality, inference_time])
                hypervolume_points_StableYolo.append(hypervolume_points)
                inference_times.append(inference_time)
                ref_point = np.array([1, 50000])
                print('ref_point: ', ref_point)
                from pymoo.indicators.hv import HV

                indicator = HV(ref_point=ref_point)

                hypervolume = indicator(np.array(hypervolume_points))
                print('Hypervolume: ', hypervolume)
                average_HY_StableYolo.append(hypervolume)

                if plot_points:
                    # plt.title('{}'.format(dir_data.replace('Data/', '').replace('.csv', '').replace('_AllNumeric', '')))
                    for point in hypervolume_points:
                        plt.scatter(point[0], point[1], color='red', label='Ref_point')
                    plt.scatter(ref_point[0], ref_point[1], color='black', label='Ref_point')
                    plt.xlabel("1-image_quality")
                    plt.ylabel("inference_time", labelpad=5)
                    # plt.legend()
average_HY_StableYolo = np.mean(average_HY_StableYolo)
average_HY_GreenStableYolo = np.mean(average_HY_GreenStableYolo)
print('average_HY_StableYolo: ', average_HY_StableYolo)
print('average_HY_GreenStableYolo: ', average_HY_GreenStableYolo)

if plot_points:
    plt.show()



#
# max_time = max(inference_times)
# min_time = min(inference_times)
#
# for temp_run in hypervolume_points_StableYolo:
#     for temp_point in temp_run:
#         temp_point[1] = (temp_point[1] - min_time) / (max_time - min_time)
#
# for temp_run in hypervolume_points_GreenStableYolo:
#     for temp_point in temp_run:
#         temp_point[1] = (temp_point[1] - min_time) / (max_time - min_time)
#
#
# ref_point = np.array([1.001, 1.001])
# print('ref_point: ', ref_point)
# from pymoo.indicators.hv import HV
# indicator = HV(ref_point=ref_point)
#
# for temp_run in hypervolume_points_GreenStableYolo:
#     hypervolume = indicator(np.array(temp_run))
#     print('GreenYollo Hypervolume: ', hypervolume)
#     average_HY_GreenStableYolo.append(hypervolume)
#
#     if plot_points:
#         # plt.title('{}'.format(dir_data.replace('Data/', '').replace('.csv', '').replace('_AllNumeric', '')))
#         for point in temp_run:
#             plt.scatter(point[0], point[1], color='blue', label='Ref_point')
#         plt.scatter(ref_point[0], ref_point[1], color='black', label='Ref_point')
#         plt.xlabel("1-image_quality")
#         plt.ylabel("inference_time", labelpad=5)
#
# for temp_run in hypervolume_points_StableYolo:
#         hypervolume = indicator(np.array(temp_run))
#         print('Hypervolume: ', hypervolume)
#         average_HY_StableYolo.append(hypervolume)
#
#         if plot_points:
#             # plt.title('{}'.format(dir_data.replace('Data/', '').replace('.csv', '').replace('_AllNumeric', '')))
#             for point in temp_run:
#                 plt.scatter(point[0], point[1], color='red', label='Ref_point')
#             plt.scatter(ref_point[0], ref_point[1], color='black', label='Ref_point')
#             plt.xlabel("1-image_quality")
#             plt.ylabel("inference_time", labelpad=5)
#                     # plt.legend()
# average_HY_StableYolo = np.mean(average_HY_StableYolo)
# average_HY_GreenStableYolo = np.mean(average_HY_GreenStableYolo)
# print('average_HY_StableYolo: ', average_HY_StableYolo)
# print('average_HY_GreenStableYolo: ', average_HY_GreenStableYolo)
#
# if plot_points:
#     plt.show()



