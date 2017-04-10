
def Rubic_cube(list):
    maximum = list.index(max(list))
    minimum = list.index(min(list))
    for i in range(len(list)):
        if i != maximum & i != minimum:
            sum += list[i]

    average = sum / 3
    return average


def main():
    list = []
    for i in range(1,6):
        time = float(input("Enter the time forperformance"+str(i)+":"))
        list.append(time)
    average = Rubic_cube(list)
    print average


main()