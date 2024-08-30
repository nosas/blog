<title>Receipt Printer: Inception</title>

# Receipt Printer - Inception

A silly project on Twitter caught my attention.
Andrew Schmelyun's video[^1] showcased a receipt printer generating physical tickets for new GitHub issues.
He wrote up a blog post[^2] about it.
I want to do something similar.

It turns out that I have a receipt printer laying around.
Specifically, I have a Star Micronics MCP31LB thermal receipt printer.
I will use it to print daily updates about myself, my projects, and my cats.

## Idea

I have a handful of services that I use daily: cat updates, GitHub issues, TODO task list, and calendar scheduling.
I'd like updates from these services to be printed on a receipt throughout the day.
In the morning, I want it to print a summary of the day.
Throughout the day, I want it to print updates regarding my cats, GitHub issues, task list completions, and whatever else I have going on (news, discord, etc.).
At the end of the day, I want it to print a summary of the day.

I asked ChatGPT to produce a sample output of the receipt's content:

> I have a receipt printer connected to my home lab. I would like for it to regularly print stuff.
> Assume it has an SDK and can communicate with local services, external APIs, etc.
>
> It starts with BEGIN \<CURRENT_DAY\>.
> It prints a time stamped event, such as weather update, calendar update, cat update, etc.
> Finally, at the end of the day it prints a summary block followed by an END \<CURRENT_DAY\> block.
>
> Can you show me a sample output for this?

After a few iterations of instructions, I got a decent starting point for the receipt's content.
It's not quite what I want, but I'll massage it out as I get stuff printing.

```log
=======================
BEGIN TUESDAY, AUGUST 27, 2024
=======================

[08:00 AM]
Weather Update:
- Temperature: 72°F
- Chance of Rain: 20%
- Wind: 5 mph N

-----------------------

[08:15 AM]
Daily Calendar Events:
- 09:00 AM: Team Meeting
- 11:00 AM: Project Review
- 02:00 PM: Doctor Appointment

-----------------------

[08:30 AM]
Cat Health Update:
- Luna: Ate breakfast at 08:25 AM.
- Max: Weight: 12 lbs. Checked at 08:20 AM.

-----------------------

[12:00 PM]
Midday Weather Update:
- Temperature: 78°F
- Chance of Rain: 10%
- Wind: 7 mph NW

-----------------------

[01:00 PM]
Reminder:
- Lunch break! Take some time to relax.

-----------------------

[05:00 PM]
GitHub Commits:
- Repo: cat-tracker
- Latest Commit: "Added health update notification system."

-----------------------

[07:00 PM]
Evening Weather Update:
- Temperature: 70°F
- Chance of Rain: 5%
- Wind: 4 mph NW

-----------------------

[09:00 PM]
Summary:
- Weather: No significant changes.
- Calendar: 3 events completed.
- Cat Updates: Luna ate 3 times, Max ate 2 times.
- GitHub: 5 commits made today.

=======================
END TUESDAY, AUGUST 27, 2024
=======================
```

---

## Implementation

My implementation will be different from Andrew's.
Where he used PHP, a Raspberry Pi, and a PHP library for communicating with the receipt printer, I will use Python, message queues, and deploy it in my Kubernetes cluster.
Overkill?
Yeah, maybe.
It should be fun, though.

This printer is awesome.
Can you believe that it has ...

1. fantastic online manual[^3]
2. open-source SDK[^4]
3. detailed protocol overview for developers[^5]

The SDK's technology stack - PHP and .NET - is not my favorite, so I will write my own in Python.
I have to remember to stay focused on the goal: I want to print stuff on a receipt printer.
The Python SDK does not have to be perfect.

Once I've converted the SDK to Python, I can create a simple web service for the print server.
Then, the fun can begin.

### Data Collection & Printing

The data to be printed is spread across many services.
I need to figure out how to collect it in a way that I can print it on the receipt printer.

At a high-level, I anticipate designing a pub-sub system.
When new updates occur (completed a task, new cat update, etc.), they are published to a topic.
Cron jobs can also publish to the topic, such as daily summaries, important news, and whatnot.
A subscriber will listen to the topic, send the print request to the printer, and mark the update as printed once the printer confirms the receipt was printed.

---

## Conclusion

I want to print stuff on a receipt printer.
I am lucky that the printer has so much documentation, so it should be a fun project.
Let's get started.

## References

[^1]: [https://x.com/aschmelyun/status/1506960015063625733](https://x.com/aschmelyun/status/1506960015063625733)
[^2]: [https://aschmelyun.com/blog/i-built-a-receipt-printer-for-github-issues/](https://aschmelyun.com/blog/i-built-a-receipt-printer-for-github-issues/)
[^3]: [https://star-m.jp/products/s_print/sdk/StarCloudPRNT/manual/en/index.html](https://star-m.jp/products/s_print/sdk/StarCloudPRNT/manual/en/index.html)
[^4]: [https://github.com/star-micronics/cloudprnt-sdk/tree/master](https://github.com/star-micronics/cloudprnt-sdk/tree/master)
[^5]: [https://star-m.jp/products/s_print/sdk/StarCloudPRNT/manual/en/protocol-reference/index.html](https://star-m.jp/products/s_print/sdk/StarCloudPRNT/manual/en/protocol-reference/index.html)