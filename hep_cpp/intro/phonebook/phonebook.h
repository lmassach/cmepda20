#ifndef PHONEBOOK_H
#define PHONEBOOK_H

#include <string>
#include <vector>
#include <algorithm>
#include <map>

/* These classes implements the "vector of containers" representation.
 * It is more OO / intuitive.
 */
class PhoneBookEntry {
  public:
    PhoneBookEntry() {}
    PhoneBookEntry(const char *new_name, const char *new_surname,
                   const char *new_number):
        name(new_name), surname(new_surname), number(new_number) {}
    PhoneBookEntry(const std::string &new_name, const std::string &new_surname,
                   const std::string &new_number):
        name(new_name), surname(new_surname), number(new_number) {}

    std::string name, surname, number;
    std::map<std::string,std::string> numbers;

    void addNumber(const std::string &type, const std::string &num) {
        numbers[type] = num;
    }
    void addNumber(const char *type, const char *num) {
        numbers[type] = num;
    }

    std::string str() const {
        //return std::format("{}, {}: {}", surname, name, number);
        std::string res(surname);
        res += ", ";
        res += name;
        res += ": ";
        res += number;
        return res;
    }

    std::string strFull() const {
        //return std::format("{}, {}: {}", surname, name, number);
        std::string res(surname);
        res += ", ";
        res += name;
        res += ": ";
        res += number;
        for (auto it = numbers.cbegin(); it != numbers.cend(); it++) {
            res += "\n  ";
            res += it->first;
            res += ": ";
            res += it->second;
        }
        return res;
    }

    bool operator<(const PhoneBookEntry &other) const {
        if (surname < other.surname) return true;
        if (surname > other.surname) return false;
        return name < other.name;
    }
};

class PhoneBookVOC {
  public:
    PhoneBookVOC() {}

    void addEntry(const PhoneBookEntry &entry) {
        //entries_.push_back(entry);
        entries_.push_back(entry);
    }
    void addEntry(const char *name, const char *surname, const char *number) {
        //entries_.push_back(PhoneBookEntry(name, surname, number));
        entries_.emplace_back(name, surname, number);
    }
    void addEntry(const std::string &name, const std::string &surname,
                  const std::string &number) {
        //entries_.push_back(PhoneBookEntry(name, surname, number));
        entries_.emplace_back(name, surname, number);
    }

    PhoneBookEntry& at(size_t index) {
        return entries_.at(index);
    }

    void sort() {
        std::sort(entries_.begin(), entries_.end());
    }

    void addPrefixes() {
        std::transform(entries_.begin(), entries_.end(), entries_.begin(),
                       addPrefix_);
    }

    std::string str() const {
        std::string res;
        for (auto it = entries_.cbegin(); it != entries_.cend(); it++) {
            res += it->strFull();
            res += '\n';
        }
        return res;
    }

  private:
    std::vector<PhoneBookEntry> entries_;

    static PhoneBookEntry addPrefix_(const PhoneBookEntry &old) {
        PhoneBookEntry res(old);
        if (!res.number.empty() && res.number.front() != '+')
            res.number.insert(0, "+39");
        for (auto it = res.numbers.begin(); it != res.numbers.end(); it++)
            if (!it->second.empty() && it->second.front() != '+')
                it->second.insert(0, "+39");
        return res;
    }
};

/* This class implements the "container of vectors" representation: each
 * instance of the class is an entire phonebook. It should be faster,
 * particularly when reading one member of each entry.
 */
class PhoneBookCOV {
  public:
    PhoneBookCOV() {}

  private:
    std::vector<std::string> name_, surname_, number_;
};


#endif // PHONEBOOK_H
