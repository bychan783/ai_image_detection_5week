"""두 정수의 사칙연산 결과를 출력하는 간단한 계산기."""


def get_integers():
    # 두 정수를 입력받는다.
    first = int(input("첫 번째 정수를 입력하세요: "))
    second = int(input("두 번째 정수를 입력하세요: "))
    return first, second


def calculate(first, second):
    # 나눗셈을 제외한 사칙연산 결과를 계산한다.
    results = {
        "+": first + second,
        "-": first - second,
        "*": first * second,
    }
    if second != 0:
        results["/"] = first / second
    return results


def print_results(results, division_possible):
    # 계산 결과와 나눗셈 오류를 출력한다.
    for operator in ("+", "-", "*"):
        print(f"{operator}: {results[operator]}")

    if division_possible:
        print(f"/: {results['/']}")
    else:
        print("/: 오류 - 0으로 나눌 수 없습니다.")


def main():
    # 입력 오류를 처리하고 계산기를 실행한다.
    try:
        first, second = get_integers()
    except ValueError:
        print("오류: 정수만 입력할 수 있습니다.")
        return

    results = calculate(first, second)
    print_results(results, second != 0)


if __name__ == "__main__":
    main()
