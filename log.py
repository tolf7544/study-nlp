def printTitle(msg: str):
    print("========================================\n")
    print(f"{msg}")
    print("\n========================================")

def printTestLog(expected: str, output: str):
        print("\t예상 출력값")
        print("\t\t",expected)
        print("\t실제 출력값")
        print("\t\t",output,"\n")

def pr_t_1(msg: str):
    print(f"\t{msg}")

def pr_t_2(msg: str):
    print(f"\t\t{msg}")

def pr_t_3(msg: str):
    print(f"\t\t\t{msg}")