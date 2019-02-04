# -*- coding: utf-8 -*-

"""
Some functions to load and convert data
"""

__author__ = 'PKU ChaiZ'


def get_comment_2_medicine(file_name):
    """
    generate a { comment-text: medicine-text } dictionary
    """
    with open(file_name, 'r') as f:
        lines = [(item.strip()) for item in f]
    comment_2_medicine = {}
    for line in lines:
        comment, medicine = tuple(line.split('###'))
        comment = comment.strip()
        medicine = medicine.strip()
        comment_2_medicine[comment] = medicine
    return comment_2_medicine


def get_medicine_2_comment(comment_2_medicine):
    """
    generate a {medicine-text: comment-text list } dictionary
    """
    medicine_2_comments = {}
    for key, value in comment_2_medicine.items():
        if value not in medicine_2_comments:
            medicine_2_comments[value] = [key]
        else:
            medicine_2_comments[value].append(key)
    return medicine_2_comments


if __name__ == '__main__':
    com2med = get_comment_2_medicine('checking_data/mem_data')
    med2com = get_medicine_2_comment(com2med)
    print(com2med)
    print(med2com)
