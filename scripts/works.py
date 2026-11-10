"""
Define the base concept of licensed material (work).
"""

import logging
from collections import namedtuple

from tabulate import tabulate

"""
Record the warnings, errors and conflicts during analysis of work
"""
Evn = namedtuple('Event', ['type', 'info'])
class EVENT(object):

    PERSONAL_OPEN_POLICY = Evn('policy', 'The desired release scenario of $WNAME$ is for PERSONAL')
    SHARE_OPEN_POLICY = Evn('policy', 'The desired release scenario of $WNAME$ is for FREE SHARING')
    SELL_OPEN_POLICY = Evn('policy', 'The desired release scenario of $WNAME$ is for SELLING')

    LICENSE_TYPE_MISMATCH_WARNING = Evn('licenses warning', 'License $LNAME$ is NO suitable for $WNAME$ ($WTYPE$) licensing')
    LICENSE_DISCLOSE_SELF_WARNING = lambda work_name, license_name: Evn('licenses warning', f"Work {work_name} must remain OPEN SOURCE (raw) or provide a READABLE COPY of source code according to license {license_name}")
    LICENSE_NO_FOUND_ERROR = lambda path : Evn("licenses error", f"License $LNAME$ could NOT be located in the parser file {path}")
    LICENSE_SHARING_PROHIBITED_ERROR = lambda work_name, license_name: Evn("licenses error", f"Work {work_name} CONNOT BE SHARED due to the prohibition of license {license_name}")
    AUX_RESULT_SHARING_PROHIBITED_ERROR = lambda result_name, source_name, license_name: Evn("licenses error", f"Result {result_name} CANNOT BE SHARED because license {license_name} of auxiliary source {source_name} does not cover redistribution of this result")
    
    MIXWORKS_NO_FOUND_WARNING = Evn('analysis warning', 'Work $WNAME$ does NOT have mixworks as expected')

    LICENSE_IN_COMPAT_ERROR = lambda work_name, original_license_name, new_license_name: Evn("analysis error", f"Work {work_name} under {original_license_name} CONNOT BE RELICENSED to license {new_license_name} due to incompatibility.")
    MULTIPLE_COPYLEFT_ERROR = lambda copylefts: Evn("analysis error", f"Work $WNAME$ has a license conflict as it involves multiple copyleft licenses: {', '.join([l.license_name for l in copylefts])}")
    
    RIGHT_NO_GRANT_WARNING = lambda work_name, right_name: Evn('rights warning', f'Work {work_name} does NOT EXPLICITLY grant you the right to {right_name.upper()}')
    RIGHT_NO_GRANT_ERROR = lambda work_name, right_name: Evn('rights error', f'Work {work_name} does NOT grant you the right to {right_name.upper()}')
    RIGHT_NO_GRANT_RELICENSE_INCOMPAT = lambda work_name, license_name, target_name: Evn(
        'rights error',
        f'Work {work_name} does NOT grant you the right to RELICENSE and license {license_name} is incompatible with {target_name}',
    )
    
    STATE_CHANGES = lambda work_name, license_name: Evn("restriction", f"Work $WNAME$ must STATE the CHANGES according to {license_name} of {work_name}")
    INCLUDE_LICENSE = lambda work_name, license_name: Evn("restriction", f"Work $WNAME$ must RETAIN the ORIGINAL LICENSE file according to {license_name} of {work_name}")
    INCLUDE_NOTICE = lambda work_name, license_name: Evn("restriction", f"Work $WNAME$ must RETAIN all NOTICE files (e.g. copyright, patent, trademark, and attribution) from the source form of {work_name} according to {license_name}")
    INCLUDE_ORIGINAL = lambda work_name, license_name: Evn("restriction", f"Work $WNAME$ must include the copies of the original work or the instructions to obtain copies of {work_name} according to {license_name}")
    INCLUDE_USE_RESTRICTION = lambda work_name, license_name: Evn("restriction", f"Work $WNAME$ must comply with the USE restriction terms according to {license_name} of {work_name}")
    INCLUDE_RUNTIME_RESTRICTION = lambda work_name, license_name: Evn("restriction", f"Work $WNAME$ must comply with the RUNTIME restriction terms according to {license_name} of {work_name}")

class Work(object):
    def __init__(self, name:str, type:str, form:str, license_name:str='TBD'):
        self.name = name
        self.type = type # software/data/model or mix (mix of works with different type)
        self.form = form # raw/binary/saas
        self.license_name = license_name # MIT, Apache-2.0, Unlicense
        self.assigned_license_name = None # Manual assigned license for this work, call analysis of license parser to take effect
        self.license = None # Need license parser to assign a license instance
        
        """
        Bundle of Works:
        +---------------++---------------+
        |   mixwork1    ||   Mixwork2    |
        +---------------++---------------+     
        """

        self.mixworks = [] # list of (works, usage), this work contain a bundle of works
        self.subworks = [] # list of (work, usage), subworks are input of this work and will be released with this work
        self.auxworks = [] # list of (work, usage), aux works will not be released with this work

        self.relied_works = [] # All relied works for this work
        
        self.caution = []
        self.determination = None
        
        # Assign 'License' instance for this work
        #self.license = self.parse_license(license_name)
        #self.parser.work_type_check(self.license, self.type)

    def __repr__(self):
        """Readable structured repr, with nested Work printing support."""

        return (
            f"Work(\n"
            f"  name={self.name!r},\n"
            f"  type={self.type!r},\n"
            f"  form={self.form!r},\n"
            f"  license_name={self.license_name!r},\n"
            f"  assigned_license_name={self.assigned_license_name!r},\n"
            f"  mixworks={self.mixworks},\n"
            f"  subworks={self.subworks},\n"
            f"  auxworks={self.auxworks},\n"
            f"  relied_works={self.relied_works!r},\n"
            # f"  caution={self.caution!r}\n"
            f")"
        )

    def set_license(self, license):
        # Assign or Reset 'License' instance to this work
        self.license = license
        self.license_name = license.short_id
    
    def is_include_mixworks(self):
        if len(self.mixworks) > 0: # At least one mixworks
            return True
        return False

    def is_registered(self):
        if self.license is None:
            return False
        return True
    
    def is_relied_work(self, work):
        if work in self.mixworks + self.subworks + self.auxworks:
            return True
        return False

    def add_event(self, event):
        if isinstance(event, list):
            self.caution += event
        if isinstance(event, Evn):
            self.caution.append(event)

    def has_relied_work(self, exclude=[]):
        coverage = []
        for _type, _works in zip(['aux', 'mix', 'sub'], [self.auxworks, self.mixworks, self.subworks]):
            if _type not in exclude:
                coverage = coverage + _works
        if len(coverage) > 0:
            return True
        return False
    
    def assign_license(self, assigned_license_name:str):
        if assigned_license_name:
            self.assigned_license_name = assigned_license_name
            return True
        return False
    
    # Recursion, you can exclude 'aux', 'mix', 'sub' works
    def find_relied_works(self, exclude=[]) -> list:
        relied_works = []
        coverage = []
        for _type, _works in zip(['aux', 'mix', 'sub'], [self.auxworks, self.mixworks, self.subworks]):
            if _type not in exclude:
                coverage = coverage + _works
        for work, usage in coverage:
            if work.has_relied_work(exclude):
                # Recursively find the relied works
                relied_works += work.find_relied_works(exclude) + [(work, usage)] # Insert new found works from left
            else:
                relied_works += [(work, usage)] 
        return relied_works
    
    # The relied works exlcude aux works will be share with this work (subworks, mixworks and this work)
    def find_shared_works(self):
        return self.find_relied_works(exclude=['aux'])
    
    # Find all mixworks of this work
    def find_mixworks(self):
        return self.find_relied_works(exclude=['aux', 'sub'])
    
    # ??? Removal of redundancies, priority of work retained: mixworks > subworks > auxworks
    # deduplicate event???
    def deduplicate_relied_works(self, exclude=[]):
        all_relied_works = {
            'mix' : self.find_relied_works(exclude=['aux', 'sub']),
            'sub' : self.find_relied_works(exclude=['aux', 'mix']),
            'aux' : self.find_relied_works(exclude=['mix', 'sub'])
        }
        return
    
    # Return the name of open policy by searching from the events in self.caution
    def find_open_policy(self):
        # Search the open policy information
        _open_policy_event = [EVENT.PERSONAL_OPEN_POLICY, EVENT.SHARE_OPEN_POLICY, EVENT.SELL_OPEN_POLICY]
        _open_policy_name = ['Personal Use', 'Share', 'Sell']
        #opevent, opname = None, None
        opevent = self.filter_events_by_type('policy')
        if opevent:
            opname = None
            for _policy, _name in zip(_open_policy_event, _open_policy_name):
                if opevent[0] == _policy: # Only compare the first open policy event
                    opname = _name
                    break
            if opname:
                return opevent[0], opname
        return None, None # It will return None if the open policy is no found
    
    # Filter the events in self.caution by its type
    def filter_events_by_type(self, type):
        finds = []
        for _event in self.caution:
            if type.casefold() in _event.type.casefold():
                finds.append(_event)
        return finds
    
    def replace_placeholder(self, event_string:str):
        for old, new in zip(['$WNAME$', '$WTYPE$', '$LNAME$'], [self.name, self.type, self.license_name]):
            event_string = event_string.replace(old, new)
        return event_string

    def _compliance_event_buckets(self):
        """与 summary() 相同的去重顺序：warning、error、restriction。"""
        warnings = self.filter_events_by_type('warning')
        errors = self.filter_events_by_type('error')
        restrictions = self.filter_events_by_type('restriction')

        def dedupe(events):
            return [event for index, event in enumerate(events) if event not in events[:index]]

        return dedupe(warnings), dedupe(errors), dedupe(restrictions)

    def _license_view(self, work=None):
        target = self if work is None else work
        license_obj = getattr(target, "license", None)
        name = license_obj.short_id if license_obj is not None else (target.license_name or "")
        folded = (name or "").casefold()
        if folded in {"", "unlicense", "unknow", "unknown"}:
            status = "unspecified"
        elif folded == "tbd":
            status = "unresolved"
        else:
            status = "named"
        return name or "", status

    def _step_edges(self):
        edges = []
        for release, pairs in (
            ("not_with_output", self.auxworks),
            ("with_output", self.subworks),
            ("with_output", self.mixworks),
        ):
            for child, usage in pairs:
                license_name, license_status = child._license_view()
                edges.append({
                    "name": child.name,
                    "type": child.type,
                    "license_name": license_name,
                    "license_status": license_status,
                    "release": release,
                    "usage": "" if usage is None else str(usage),
                    "produced": child._has_step_inputs(),
                    "work": child,
                })
        return edges

    def _has_step_inputs(self):
        return bool(self.auxworks or self.subworks or self.mixworks)

    def _operation_name(self, edges):
        usages = [edge["usage"] for edge in edges]
        for name in (
            "embed",
            "generate",
            "train",
            "modify",
            "amalgamate",
            "combine_mix",
            "combine",
            "distill",
            "copy",
            "use",
        ):
            if name in usages:
                return name
        return usages[0] if usages else ""

    def _collect_steps(self, seen):
        if id(self) in seen or not self._has_step_inputs():
            return []
        seen.add(id(self))
        edges = self._step_edges()
        steps = []
        for edge in edges:
            steps.extend(edge["work"]._collect_steps(seen))
        license_name, license_status = self._license_view()
        step = {
            "name": self.name,
            "type": self.type,
            "form": self.form,
            "license_name": license_name,
            "license_status": license_status,
            "operation": self._operation_name(edges),
            "inputs": [{key: value for key, value in edge.items() if key != "work"} for edge in edges],
        }
        if self.determination:
            step["determination"] = self.determination
        steps.append(step)
        return steps

    def report_payload(self) -> dict:
        """供报告使用的步骤和事件。不替代 summary() 的英文文本。"""
        warnings, errors, restrictions = self._compliance_event_buckets()

        def pack(bucket, events, prefix):
            rows = []
            for index, event in enumerate(events, 1):
                rows.append({
                    "id": f"{prefix}-{index}",
                    "bucket": bucket,
                    "label": f"{event.type.title()} - {index}",
                    "text": self.replace_placeholder(event.info),
                })
            return rows

        license_name, license_status = self._license_view()
        return {
            "engine_ran": True,
            "new_work": {
                "name": self.name,
                "type": self.type,
                "form": self.form,
                "license_name": license_name,
                "license_status": license_status,
            },
            "steps": self._collect_steps(set()),
            "events": (
                pack("warning", warnings, "W")
                + pack("error", errors, "E")
                + pack("restriction", restrictions, "R")
            ),
        }


    # Summary of this work, including license information, conflicts and restrictions
    def summary(self, save_path=None):
        lines = []

        # The basic information of this work
        work_data = [
            ["Work Name",       self.name],
            ["Type & Form",     self.type.title() + ', ' + self.form.title()],
            ["License Name",    self.license.short_id if self.license else self.license_name],
            ["Relied Work (License)",    ', '.join([f"{w.name}({w.license_name})" for w,_ in self.find_relied_works()])]
        ]

        # Search the open policy information
        opevent, opname = self.find_open_policy()
        if opname is None:
            logging.warning(f"Didn't find the open policy of Work {self.name}")
        else:
            print(f"{self.replace_placeholder(opevent.info)}.")

        warnings, errors, restrictions = self._compliance_event_buckets()
        restrictions_data = []
        for events in [warnings, errors, restrictions]:
            for idx, e in enumerate(events):
                restrictions_data.append([f"{e.type.title()} - {idx+1}", self.replace_placeholder(e.info)])

        # Summary the basic information of this work
        lines.append("")
        lines.append(f"The base information of {self.name}:")
        lines.append(tabulate(work_data, tablefmt="plain"))
        lines.append("")

        # Print open policy information
        lines.append(
            f"To {opname} this work in compliance, you must take into account "
            f"the following Warnings({len(warnings)}), Errors({len(errors)}), Restrictions({len(restrictions)})."
        )
        lines.append(tabulate(restrictions_data, tablefmt="plain"))
        lines.append("")

        # Print the license information of this work
        lines.append(f"The information of {self.license_name} license of this work is:")
        final_text = "\n".join(lines)

        # Decide output
        if save_path:
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(final_text)
        else:
            print(final_text)

        self.license.summary()

        return
