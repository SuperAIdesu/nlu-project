let haystack = [3, 3, 3, 5, 5, 5, 7, 7, 7, 9, 9, 9]
let indices = [0, 1, 2, 0, 2, 4, 0, 3, 6, 0, 4, 8]


$haystack | enumerate | each {
    |h| python src/experiment_2step.py --haystack $h.item --index ($indices | get $h.index)
}