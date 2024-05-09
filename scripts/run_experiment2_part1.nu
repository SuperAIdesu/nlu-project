let haystack = [25, 25, 25, 31, 31, 31, 35, 35, 35, 41, 41, 41]
let indices = [0, 12, 24, 0, 15, 30, 0, 17, 34, 0, 20, 40]

$haystack | enumerate | each {
    |h| python src/experiment_2step.py --haystack $h.item --index ($indices | get $h.index)
}

$haystack | enumerate | each {
    |h| python src/experiment_2step.py --haystack $h.item --index ($indices | get $h.index) --multi_retain_only
}