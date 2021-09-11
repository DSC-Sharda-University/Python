def sum_digit(num:int)->int:
    result = 0
    if num <= 0:
        return 1
    while num > 0:
        remainder = num % 10
        result = result + remainder
        num //= 10
    return result

    
print(sum_digit(123))